
import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from core.configs import DATASET_FOLDERS, DATASET_FILES
from core.datasets.utils import expand_to_aspect_ratio, resize_image, get_example
from core.constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from torchvision.transforms import Normalize
import random

import trimesh
from core.utils.renderer_pyrd import Renderer

"Part of the code has been taken from "
"4DHumans: https://github.com/shubham-goel/4D-Humans"

from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from core.configs import dataset_config
from core.datasets import DataModule
from core.datasets.soccer_dataset import SoccerDataModule
from core.utils.pylogger import get_pylogger
from core.utils.misc import task_wrapper, log_hyperparameters
from pytorch_lightning.strategies import DDPStrategy
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
torch.set_float32_matmul_precision('medium')
torch.manual_seed(0)


class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, is_train=True):
        super(DatasetTrain, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.normalize_img = Normalize(mean=cfg.MODEL.IMAGE_MEAN,
                                    std=cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.img_dir = DATASET_FOLDERS[dataset] 
        self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        self.imgname = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        if 'pose_cam' in self.data:
            self.body_pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24 
        elif 'pose' in self.data:
            self.body_pose = self.data['pose'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24 
        else:
            self.body_pose = np.zeros((len(self.imgname), NUM_PARAMS_SMPL*3), dtype=np.float32)  

        if self.body_pose.shape[1] == NUM_PARAMS_SMPL:
            self.body_pose = self.body_pose.reshape(-1,NUM_PARAMS_SMPL*3)      
        self.betas = self.data['shape'].astype(np.float)[:,:NUM_BETAS] 
        self.cam_int = self.data['cam_int']
        self.keypoints = self.data['gtkps'][:,:NUM_JOINTS]
        self.length = self.scale.shape[0]
        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext']
        else:
            self.cam_ext = np.zeros((self.imgname.shape[0], 4, 4))

        #Only for BEDLAM and AGORA
        if 'trans_cam' in self.data:
            self.trans_cam = self.data['trans_cam']
        else:
            self.trans_cam = np.zeros((self.imgname.shape[0],3))
        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        from core.constants import SMPL_MODEL_PATH
        import smplx
        self.body_model = smplx.SMPL(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)


    def get_output_mesh(self, params, cam_trans):
        print(params)
        # smpl_output = self.body_model(**{k: v for k, v in params.items()})
        params = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in params.items()}
        smpl_output = self.body_model(**params)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        return pred_vertices, pred_keypoints_3d, cam_trans


    def view_ground_truth(self, f, i, img_path, smpl_params, cam_trans, img_w, img_h, output_img_folder="./test_get_output_mesh/out/"):
        img_cv2 = cv2.imread(str(img_path))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}.obj')


        output_vertices, output_joints, output_cam_trans = self.get_output_mesh(smpl_params, cam_trans)

        mesh = trimesh.Trimesh(output_vertices[0].cpu().detach().numpy() , self.body_model.faces,
                        process=False)
        mesh.export(mesh_fname)

        # Render overlay
        # print(f)
        focal_length = (f, f)
        pred_vertices_array = (output_vertices + torch.from_numpy(output_cam_trans[:3]).to(self.device)).detach().cpu().numpy()
        print(pred_vertices_array)
        renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.body_model.faces, same_mesh_color=True)
        front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())
        final_img = front_view
        # Write overlay
        cv2.imwrite(overlay_fname, final_img)
        renderer.delete()

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        orig_keypoints_2d = self.keypoints[index].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            #Todo raise proper error
            breakpoint()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        imgname = os.path.join(self.img_dir, self.imgname[index])
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())

        item['pose'] = self.body_pose[index]
        item['betas'] = self.betas[index]

        smpl_params = {'global_orient': self.body_pose[index][:3].astype(np.float32),
                    'body_pose': self.body_pose[index][3:].astype(np.float32),
                    'betas': self.betas[index].astype(np.float32)
                    }
        item['smpl_params'] = smpl_params
        item['translation'] = self.cam_ext[index][:, 3]
        if 'trans_cam' in self.data.files:
            item['translation'][:3] += self.trans_cam[index]
        img_patch_rgba = None
        img_patch_cv = None
        img_patch_rgba, \
        img_patch_cv,\
        keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(imgname,
                                      center_x, center_y,
                                      bbox_size, bbox_size,
                                      keypoints_2d,
                                      FLIP_KEYPOINT_PERMUTATION,
                                      self.IMG_SIZE, self.IMG_SIZE,
                                      self.MEAN, self.STD, self.is_train, augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=self.use_skimage_antialias,
                                      border_mode=self.border_mode,
                                      dataset=self.dataset
                                      )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
        item['cam_int'] = np.array(self.cam_int[index]).astype(np.float32)
        item['img_disp'] = img_patch_cv
        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset

        img_h, img_w = item['img_size']

        print(f"Viewing ground truth {index}")
        self.view_ground_truth(item['cam_int'][0, 0], index, imgname, item['smpl_params'], item['translation'], img_w, img_h)


        return item

    def __len__(self):
        return int(len(self.imgname))

@hydra.main(version_base="1.2", config_path=str(root/"core/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    t = DatasetTrain(cfg, 'coco-train', is_train=True)
    # t.__getitem__(0)
    t.__getitem__(1)
    t.__getitem__(2)
    t.__getitem__(3)
    # t.__getitem__(4)


if __name__ == "__main__":
    main()
        
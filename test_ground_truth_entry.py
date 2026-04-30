
import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
# from ..configs import DATASET_FOLDERS, DATASET_FILES
from core.datasets.utils import resize_image, get_example
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

import torch
import numpy as np

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Parameters
    ----------
    axis_angle : torch.Tensor, shape (..., 3)
        Axis-angle representation where the magnitude of the vector
        is the angle of rotation in radians.
    
    Returns
    -------
    torch.Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    batch_shape = axis_angle.shape[:-1]
    
    # Compute angle (magnitude of the axis-angle vector)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)          # (..., 1)
    
    # Normalize to get unit axis (handle near-zero angles safely)
    axis = axis_angle / (angle + 1e-8)                            # (..., 3)
    
    # Components
    x = axis[..., 0:1]   # (..., 1)
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    
    cos = torch.cos(angle)        # (..., 1)
    sin = torch.sin(angle)        # (..., 1)
    one_minus_cos = 1 - cos       # (..., 1)

    # Rodrigues' rotation formula:
    # R = cos(θ)I + sin(θ)[k]× + (1 - cos(θ))kkᵀ
    #
    # where [k]× is the skew-symmetric cross-product matrix of axis k:
    # [k]× = [[ 0, -z,  y],
    #         [ z,  0, -x],
    #         [-y,  x,  0]]

    R = torch.stack([
        cos + x*x*one_minus_cos,    x*y*one_minus_cos - z*sin,  x*z*one_minus_cos + y*sin,
        y*x*one_minus_cos + z*sin,  cos + y*y*one_minus_cos,    y*z*one_minus_cos - x*sin,
        z*x*one_minus_cos - y*sin,  z*y*one_minus_cos + x*sin,  cos + z*z*one_minus_cos,
    ], dim=-1)                                                    # (..., 9)

    return R.reshape(*batch_shape, 3, 3)                          # (..., 3, 3)

def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation using the inverse
    of Rodrigues' formula.

    Parameters
    ----------
    R : torch.Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    torch.Tensor, shape (..., 3)
        Axis-angle representation where the magnitude of the vector
        is the angle of rotation in radians.
    """
    batch_shape = R.shape[:-2]

    # --- Step 1: Extract angle from trace ---
    # trace(R) = 1 + 2*cos(θ)  →  θ = arccos((trace - 1) / 2)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]          # (...,)
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)           # numerical safety
    angle = torch.acos(cos_angle)                                  # (...,)

    # --- Step 2: Extract axis from skew-symmetric part ---
    # R - Rᵀ = 2*sin(θ) * [k]×
    # so the axis components live in the off-diagonals:
    #   [k]× = [[ 0, -z,  y],
    #           [ z,  0, -x],
    #           [-y,  x,  0]]
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],   # 2*sin(θ)*x
        R[..., 0, 2] - R[..., 2, 0],   # 2*sin(θ)*y
        R[..., 1, 0] - R[..., 0, 1],   # 2*sin(θ)*z
    ], dim=-1)                                                      # (..., 3)

    # --- Step 3: Normalize axis ---
    # axis = 2*sin(θ)*k  →  divide by 2*sin(θ) to get unit k
    sin_angle = torch.sin(angle).unsqueeze(-1)                     # (..., 1)
    axis = axis / (2.0 * sin_angle + 1e-8)                        # (..., 3)

    # --- Step 4: Axis-angle = unit_axis * angle ---
    axis_angle = axis * angle.unsqueeze(-1)                        # (..., 3)

    # --- Step 5: Handle edge cases ---

    # Case A: angle ≈ 0  →  no rotation, return zero vector
    # (axis is undefined but axis_angle → 0 naturally via small angle)
    near_zero = (angle < 1e-6)                                     # (...,)

    # Case B: angle ≈ π  →  sin(θ) ≈ 0, skew-symmetric part vanishes
    # Must extract axis from R + Rᵀ = 2*cos(θ)I + 2*kkᵀ
    # → kkᵀ = (R + Rᵀ - 2cos(θ)I) / 2(1 + cos(θ))  ... but simpler:
    # diagonal of (R + I)/2 = (1 + cos(θ) + k_i²(1-cos(θ))) / 2
    # at θ=π: diag = k_i²  →  just take sqrt of diagonal
    near_pi = (angle > (torch.pi - 1e-6))                         # (...,)

    if near_pi.any():
        # (R + I) / 2 at θ=π gives kkᵀ, diagonal = k_i²
        RpI = (R + torch.eye(3, device=R.device, dtype=R.dtype))  # (..., 3, 3)
        axis_pi = torch.stack([
            RpI[..., 0, 0],
            RpI[..., 1, 1],
            RpI[..., 2, 2],
        ], dim=-1).clamp(min=0.0).sqrt()                           # (..., 3)

        # Recover signs from off-diagonals
        # R[2,1] - R[1,2] sign tells us sign of x (even if magnitude ≈ 0)
        axis_pi = axis_pi * torch.sign(torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1) + 1e-8)                                         # (..., 3)

        axis_angle_pi = axis_pi * torch.pi                         # (..., 3)
        axis_angle = torch.where(
            near_pi.unsqueeze(-1), axis_angle_pi, axis_angle
        )

    axis_angle = torch.where(
        near_zero.unsqueeze(-1), torch.zeros_like(axis_angle), axis_angle
    )

    return axis_angle                                              # (..., 3)

class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, img_dir, game_name, poses_npz_path, camera_npz_path):
        super(DatasetTrain, self).__init__()

        self.dataset = dataset
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

        self.img_dir = img_dir
        self.pose_data = np.load(poses_npz_path, allow_pickle=True)
        self.camera_data = np.load(camera_npz_path, allow_pickle=True)
        # print(self.camera_data)
        
        # self.img_dir = DATASET_FOLDERS[dataset] 
        # self.data = np.load(DATASET_FILES[is_train][dataset], allow_pickle=True)
        # self.imgname = self.data['imgname']
        self.imgname_prefix = game_name
        # self.scale = self.data['scale']
        self.scale = 1.0
        # self.center = self.data['center']
        self.center = self.pose_data['transl']
        # if 'pose_cam' in self.data:
        #     self.body_pose = self.data['pose_cam'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24 
        # elif 'pose' in self.data:
        #     self.body_pose = self.data['pose'][:, :NUM_PARAMS_SMPL*3].astype(np.float) #Change 24 
        # if 'body_pose' in self.pose_data:
        self.body_pose = self.pose_data['body_pose'].astype(np.float)
        self.num_players = self.body_pose.shape[0]
        self.num_frames = self.body_pose.shape[1]
        # else:
        #     self.body_pose = np.zeros((len(self.imgname), NUM_PARAMS_SMPL*3), dtype=np.float32)  

        # if self.body_pose.shape[1] == NUM_PARAMS_SMPL:
        #     self.body_pose = self.body_pose.reshape(-1,NUM_PARAMS_SMPL*3)      
        # self.betas = self.data['shape'].astype(np.float)[:,:NUM_BETAS] 
        self.betas = self.pose_data['betas'].astype(np.float)[:,:NUM_BETAS] # shape (N, 10)
        self.global_orients = self.pose_data['global_orient'].astype(np.float) # (N, T, 3)

        # self.cam_int = self.data['cam_int']
        self.cam_int = self.camera_data['K'] # shape (T, 3, 3), T is number frames

        # self.keypoints = self.data['gtkps'][:,:NUM_JOINTS]
        # self.length = self.scale.shape[0]
        # if 'cam_ext' in self.data:
        #     self.cam_ext = self.data['cam_ext']
        # else:

        # Cam extrinsic matrix is a 3 x 4 matrix (3 x 3 rotation + 3 x 1 translation)
        self.cam_ext = np.zeros((self.body_pose.shape[1], 3, 4)) # (T, 4, 4)
        self.cam_ext[:, :3, :3] = self.camera_data['R']
        self.cam_ext[:, :3, 3] = self.camera_data['t']

        self.cam_distort = self.camera_data['k']

        #Only for BEDLAM and AGORA
        # if 'trans_cam' in self.data:
        #     self.trans_cam = self.data['trans_cam']
        # else:
        #     self.trans_cam = np.zeros((self.imgname.shape[0],3))

        # if 't' in self.camera_data:
        #     self.trans_cam = self.camera_data['t']
        # else:
        #     self.trans_cam = None
        
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        from core.constants import SMPL_MODEL_PATH
        import smplx
        self.body_model = smplx.SMPL(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        
        log.info(f'Loaded {game_name} dataset, num players {self.num_players}, num frames {self.num_frames}')

    def get_frame_and_player(self, index, num_frames):
        # Player, frame
        return index // num_frames, index % num_frames
    
    # def project_points(self, X_world, R, t, K):
    #     # World → Camera
    #     X_cam = (R @ X_world.T).T + t

    #     print("Camera: ", X_cam)
    #     X = X_cam[0]
    #     Y = X_cam[1]
    #     Z = X_cam[2]

    #     # Perspective divide
    #     x = X / Z
    #     y = Y / Z

    #     # Apply intrinsics
    #     u = K[0,0] * x + K[0,2] # focal point x is K[0, 0], cx is K[0,2]
    #     v = K[1,1] * y + K[1,2] # focal point y is K[1, 1], cy is K[1,2]

    #     return u, v

    def project_points(self, obj_pts, R, t, f, principal_points, k, *args, **kwargs):
        # world to camera transformation
        pts_c = obj_pts @ R.T + t
        img_pts = pts_c[..., :2] / pts_c[..., 2:]

        # here we add a small hack to make sure the points are in the image
        r = np.square(img_pts).sum(-1, keepdims=True)

        # we assume the distortion will only have minor impact on the projection
        r = np.clip(r, 0, 0.5 / min(max(np.abs(k).max(), 1), 1))

        # if k.shape[0] <= 2:
        #     img_pts = img_pts * (1 + k[0] * r + k[1] * np.square(r))
        # else:
        #     img_pts = img_pts * (1 + k[0] * r + k[1] * np.square(r) + k[2] * np.power(r, 3))
        d = np.ones_like(r)
        for i in range(0, k.shape[0]):
            d = d + k[i] * np.power(r, i + 1)
        img_pts = img_pts * d
        img_pts = img_pts * f + principal_points[None, :]
        return img_pts

    def get_output_mesh(self, params, cam_trans, rotation_matrix):

        print(params)
        # smpl_output = self.body_model(**{k: v for k, v in params.items()})
        params = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in params.items()}

        R_existing = axis_angle_to_matrix(params['global_orient'])

        R_new = torch.tensor(rotation_matrix, dtype=torch.float32, device=R_existing.device) @ R_existing

        global_orient_aa = matrix_to_axis_angle(R_new)

        params['global_orient'] = global_orient_aa

        smpl_output = self.body_model(**params, pose2rot=True)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        return pred_vertices, pred_keypoints_3d, cam_trans


    def view_ground_truth(self, f, i, img_path, smpl_params, cam_trans, rotation, player_center, k, img_w, img_h, output_img_folder="./test_get_output_mesh/out/"):
        img_cv2 = cv2.imread(str(img_path))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}{img_ext}')
        mesh_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}.obj')


        output_vertices, output_joints, output_cam_trans = self.get_output_mesh(smpl_params, cam_trans, rotation)

        mesh = trimesh.Trimesh(output_vertices[0].cpu().detach().numpy() , self.body_model.faces,
                        process=False)
        mesh.export(mesh_fname)

        vertices = output_vertices
        t = torch.tensor(cam_trans).to(vertices.device) 
        k = torch.tensor(k).to(vertices.device)
        # .to(vertices.device)
        R = rotation
        # .to(vertices.device)
        
        vertices_cam = vertices
        # vertices_cam = (R[:, None] @ vertices[..., None]).squeeze(-1)
        vertices_cam += t
        vertices_normalized = vertices_cam[..., :2] / vertices_cam[..., 2:]
        r = vertices_normalized.square().sum(-1, keepdims=True)
        # r.clamp_(0, 1)
        k1 = k[0:1]
        k2 = k[1:2]

        scale = 1 + k1 * r + k2 * r.square()
        vertices_cam[:2] *= scale

        # transform back to world coordinates
        vertices_cam -= t
        # vertices_cam = (R[:, None].transpose(-1, -2) @ vertices_cam[..., None]).squeeze(-1)
        vertices = vertices_cam
        
        output_vertices = vertices
        # center_xy = [player_center[:2]]
        
        # # Normalize to image coords (subtract principal point, divide by focal)
        # player_center_norm = (center_xy - np.array([img_w / 2, img_h / 2])) / f  # (1, 2)

        # # Step 2: Undistort — invert the radial distortion iteratively
        # def undistort_points(img_pts, k, iters=10):
        #     """Iteratively undistort normalized image points."""
        #     pts = img_pts.copy()
        #     for _ in range(iters):
        #         r = np.square(pts).sum(-1, keepdims=True)
        #         r = np.clip(r, 0, 0.5 / min(max(np.abs(k).max(), 1), 1))
        #         d = np.ones_like(r)
        #         for i in range(k.shape[0]):
        #             d = d + k[i] * np.power(r, i + 1)
        #         pts = img_pts / d   # invert the scaling
        #     return pts

        # player_center_undistorted = undistort_points(player_center_norm, k)  # (1, 2)

        # # Step 3: Re-scale back to pixel space
        # player_center_corrected_px = (
        #     player_center_undistorted * f
        #     + np.array([img_w / 2, img_h / 2])
        # )

        # corrected_center = np.array([
        #     player_center_corrected_px[0, 0],
        #     player_center_corrected_px[0, 1],
        #     player_center[2]   # keep original Z depth
        # ])

        transl_world = (torch.from_numpy(rotation @ player_center).to(self.device).unsqueeze(-1)).squeeze(-1)

        # Render overlay
        # print(f)
        focal_length = (f, f)
        pred_vertices_array = (output_vertices + torch.from_numpy(output_cam_trans).to(self.device) + transl_world).detach().cpu().numpy()
        print(pred_vertices_array)
        renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h, faces=self.body_model.faces, same_mesh_color=True)
        front_view = renderer.render_front_view(pred_vertices_array, bg_img_rgb=img_cv2.copy())
        final_img = front_view
        # Write overlay
        cv2.imwrite(overlay_fname, final_img)
        renderer.delete()


    def __getitem__(self, index):
        #player, frame = self.get_frame_and_player(index, self.body_pose.shape[1])
        # player = 8 # 1
        player = 15 # 0
        # player = 2 # 2
        # player = 4 # 3
        # player = 16 # 4
        frame = 0
        item = {}
        scale = self.scale
        center = self.center[player][frame].copy()

        num_players = self.body_pose.shape[0]
        num_frames = self.body_pose.shape[1]
        
        while (np.isnan(np.array(center)).any()):
            player = random.randint(0, num_players - 1)
            frame = random.randint(0, num_frames - 1)
            center = self.center[player][frame].copy()
        
        # ???
        # keypoints_2d = self.keypoints[index].copy()
        # orig_keypoints_2d = self.keypoints[index].copy()
        
        # center_x = center[0]
        # center_y = center[1]
        camera_rot = self.cam_ext[frame][:, :3]
        camera_trans = self.cam_ext[frame][:, 3]
        # print(self.cam_ext)
        world_coords = np.array(center)
        calib = {
            "R": camera_rot,
            "t": camera_trans,
            "k": self.cam_distort[frame],
            "f": self.cam_int[frame][0, 0],
            "principal_points": self.cam_int[frame][:2, 2],
        }
        # print("world coords", world_coords)
        # print("Calib", calib)
        #scenter_x, center_y = self.project_points(world_coords, camera_rot, camera_trans, self.cam_int[frame])
        projection = self.project_points(world_coords, **calib)
        # print(projection)
        center_x = projection[0][0]
        center_y = projection[0][1]
        # print("Center x, center y: ", center_x, center_y)
        # bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        bbox_size = 200 # Placeholder, used in augmenting image but turned off
        # Will give a bounding box size of 200px
        # print(f"Bounding box size: {bbox_size}")
        if bbox_size < 1:
            #Todo raise proper error
            breakpoint()

        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        # imgname = None
        imgname = os.path.join(self.img_dir, self.imgname_prefix, f"{frame:06d}.jpg")

        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        item['img_full_resized'] = self.normalize_img(torch.from_numpy(img_full_resized).float())

        item['pose'] = self.body_pose[player][frame]
        item['betas'] = self.betas[player]

        smpl_params = {'global_orient': self.global_orients[player][frame].astype(np.float32),
                    'body_pose': self.body_pose[player][frame].astype(np.float32),
                    'betas': self.betas[player].astype(np.float32)
                    }
        item['smpl_params'] = smpl_params
        item['translation'] = self.cam_ext[frame][:, 3]
        # item['translation'] = center

        item['rotation'] = self.cam_ext[frame][:, :3]
        

        ## TODO: Augmentations have cropping logic - however, we don't have scaling factor - we only have depth with respect to camera
        # Given in field coordinates
        # if self.trans_cam:
        #     item['translation'][:3] += self.trans_cam[index]
        img_patch_rgba = None
        img_patch_cv = None
        img_patch_rgba, \
        img_patch_cv,\
        keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(imgname,
                                      center_x, center_y,
                                      bbox_size, bbox_size,
                                      None, #keypoints_2d,
                                      FLIP_KEYPOINT_PERMUTATION,
                                      self.IMG_SIZE, self.IMG_SIZE,
                                      self.MEAN, self.STD, 
                                        False, # do_augment
                                    augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=self.use_skimage_antialias,
                                      border_mode=self.border_mode,
                                      dataset=self.dataset
                                      )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
        item['cam_int'] = np.array(self.cam_int[frame]).astype(np.float32)
        item['img_disp'] = img_patch_cv
        item['img'] = img_patch
        # print("Image: ", img_patch)
        # item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        # item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = new_center
        item['box_size'] = bbox_w * scale_aug
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset
        item['gender']  = 0

        item['player_center'] = center # same to box_center
        item['cam_distortion'] = self.cam_distort[frame]

        img_h, img_w = item['img_size']

        print(f"Viewing ground truth {index}")
        self.view_ground_truth(item['cam_int'][0, 0], index, imgname, item['smpl_params'], 
                                item['translation'], item['rotation'], item['player_center'], item['cam_distortion'], img_w, img_h)
        return item

    def __len__(self):
        num_players = self.body_pose.shape[0]
        num_frames = self.body_pose.shape[1]
        return int(num_frames) * int(num_players)


@hydra.main(version_base="1.2", config_path=str(root/"core/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    t = DatasetTrain(cfg, 'aa', "./test_get_output_mesh/outputs/", "ARG_CRO_234105", "./test_get_output_mesh/poses/ARG_CRO_234105.npz","./test_get_output_mesh/cameras/ARG_CRO_234105.npz")
    t.__getitem__(0)
    # t.__getitem__(1)
    # t.__getitem__(2)
    # t.__getitem__(3)
    # t.__getitem__(4)


if __name__ == "__main__":
    main()
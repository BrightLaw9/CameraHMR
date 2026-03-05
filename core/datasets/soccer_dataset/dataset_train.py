
import os
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from ...utils.pylogger import get_pylogger
# from ..configs import DATASET_FOLDERS, DATASET_FILES
from ...constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from ..utils import expand_to_aspect_ratio, get_example, resize_image
from torchvision.transforms import Normalize
log = get_pylogger(__name__)


class DatasetTrain(Dataset):
    def __init__(self, cfg, dataset, img_dir, game_name, poses_npz_path, camera_npz_path, is_train=True):
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

        self.img_dir = img_dir
        self.pose_data = np.load(poses_npz_path, allow_pickle=True)
        self.camera_data = np.load(camera_npz_path, allow_pickle=True)
        print(self.camera_data)
        
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

    def __getitem__(self, index):
        player, frame = self.get_frame_and_player(index, self.body_pose.shape[1])
        item = {}
        scale = self.scale
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
        #scenter_x, center_y = self.project_points(world_coords, camera_rot, camera_trans, self.cam_int[frame])
        projection = self.project_points(world_coords, **calib)
        print(projection)
        center_x = projection[0][0]
        center_y = projection[0][1]
        print("Center x, center y: ", center_x, center_y)
        # bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        bbox_size = 200 # Placeholder, used in augmenting image but turned off
        # Will give a bounding box size of 200px
        print(f"Bounding box size: {bbox_size}")
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
        return item

    def __len__(self):
        num_players = self.body_pose.shape[0]
        num_frames = self.body_pose.shape[1]
        return int(num_frames) * int(num_players)
        
       
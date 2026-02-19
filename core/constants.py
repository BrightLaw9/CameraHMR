CHECKPOINT_PATH='data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt'
#CHECKPOINT_PATH="logs/train/runs/train_run1/checkpoints/epoch=104-step=520000.ckpt"
CHECKPOINT_PATH_SMPLX='data/pretrained-models//bedlam_v1_v2.ckpt'
CAM_MODEL_CKPT='data/pretrained-models/cam_model_cleaned.ckpt'
DENSEKP_CKPT='data/pretrained-models/densekp.ckpt'
SMPL_MEAN_PARAMS_FILE='data/smpl_mean_params.npz'
SMPL_MODEL_PATH='data/models/SMPL/SMPL_NEUTRAL.pkl'
SMPLX_MODEL_PATH='data/models/SMPLX/SMPLX_NEUTRAL.npz'

DETECTRON_CKPT='data/pretrained-models/model_final_f05665.pkl'
DETECTRON_CFG='core/utils/cascade_mask_rcnn_vitdet_h_75ep.py'
TRANSFORMER_DECODER={'depth': 6,
                    'heads': 8,
                    'mlp_dim': 1024,
                    'dim_head': 64,
                    'dropout': 0.0,
                    'emb_dropout': 0.0,
                    'norm': 'layer',
                    'context_dim': 1280}

IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
NUM_POSE_PARAMS = 23
NUM_BETAS = 10
NUM_BETAS_SMPLX = 16
NUM_JOINTS = 44
NUM_PARAMS_SMPL = 24
NUM_PARAMS_SMPLX = 24 # Only body
NUM_JOINTS_SMPLX = 44
NUM_PARAMS_SMPLX_HANDS = 30
NUM_DENSEKP_SMPL = 138
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]
SMPL_to_J19 = 'data/train-eval-utils/SMPL_to_J19.pkl'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/train-eval-utils/J_regressor_extra.npy'
SMPLX2SMPL='data/train-eval-utils/smplx2smpl.pkl'
DOWNSAMPLE_MAT='data/train-eval-utils/downsample_mat.pkl'
REGRESSOR_H36M='data/train-eval-utils/J_regressor_h36m.npy'
#SMPLX_MODEL_DIR='data/models/smplx_neutral_head/models_lockedhead/smplx'
SMPLX_MODEL_DIR='data/models/SMPLX'
SMPL_MODEL_DIR='data/models/SMPL'
VITPOSE_BACKBONE='data/train-eval-utils/vitpose_backbone.pth'
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

MANO_LEFT_MEAN_6D = [
    0.9137306 ,  0.40589133,  0.01867034, -0.40117565,  0.9084971 ,
    -0.11700923,  0.72573376,  0.68083405, -0.09887145, -0.68766904,
    0.7221707 , -0.07470501,  0.9782054 ,  0.19115911, -0.0810706 ,
    -0.18243419,  0.97769463,  0.10407168,  0.86194086,  0.50082207,
    0.07896362, -0.50669086,  0.8564043 ,  0.09917654,  0.7607176 ,
    0.6414799 ,  0.09905688, -0.6490796 ,  0.7522939 ,  0.11291368,
    0.9392709 ,  0.33190092, -0.08724636, -0.3301477 ,  0.9433083 ,
    0.0342334 ,  0.7963314 ,  0.6034986 , -0.04056842, -0.51846385,
    0.7155929 ,  0.46810475,  0.8726636 ,  0.4882266 , -0.00964513,
    -0.47441906,  0.85233253,  0.22012673,  0.9950805 ,  0.08858123,
    -0.04436439, -0.0665417 ,  0.9293623 ,  0.36312255,  0.8016535 ,
    0.59668416, -0.03632812, -0.58493584,  0.79550135,  0.15820135,
    0.80713856,  0.5890953 ,  0.03865364, -0.57224864,  0.7646009 ,
    0.29650798,  0.92279893,  0.37322533, -0.09562963, -0.3503229 ,
    0.91611946,  0.19493324,  0.9602583 ,  0.18978268,  0.20466185,
    0.03040025,  0.6577811 , -0.75259537,  0.99828583, -0.05717421,
    0.01250997,  0.04417812,  0.87632644,  0.47968763,  0.96167034,
    0.27268896, -0.02882456, -0.24881549,  0.8236024 , -0.50967634
]

MANO_RIGHT_MEAN_6D = [
    0.9137306 , -0.40589133, -0.01867034,  0.40117565,  0.9084971 ,
    -0.11700923,  0.72573376, -0.68083405,  0.09887145,  0.68766904,
    0.7221707 , -0.07470501,  0.9782054 , -0.19115911,  0.0810706 ,
    0.18243419,  0.97769463,  0.10407168,  0.86194086, -0.50082207,
    -0.07896362,  0.50669086,  0.8564043 ,  0.09917654,  0.7607176 ,
    -0.6414799 , -0.09905688,  0.6490796 ,  0.7522939 ,  0.11291368,
    0.9392709 , -0.33190092,  0.08724636,  0.3301477 ,  0.9433083 ,
    0.0342334 ,  0.7963314 , -0.6034986 ,  0.04056842,  0.51846385,
    0.7155929 ,  0.46810475,  0.8726636 , -0.4882266 ,  0.00964513,
    0.47441906,  0.85233253,  0.22012673,  0.9950805 , -0.08858123,
    0.04436439,  0.0665417 ,  0.9293623 ,  0.36312255,  0.8016535 ,
    -0.59668416,  0.03632812,  0.58493584,  0.79550135,  0.15820135,
    0.80713856, -0.5890953 , -0.03865364,  0.57224864,  0.7646009 ,
    0.29650798,  0.92279893, -0.37322533,  0.09562963,  0.3503229 ,
    0.91611946,  0.19493324,  0.9602583 , -0.18978268, -0.20466185,
    -0.03040025,  0.6577811 , -0.75259537,  0.99828583,  0.05717421,
    -0.01250997, -0.04417812,  0.87632644,  0.47968763,  0.96167034,
    -0.27268896,  0.02882456,  0.24881549,  0.8236024 , -0.50967634
]

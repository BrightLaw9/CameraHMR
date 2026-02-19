import numpy as np
import cv2
from pathlib import Path
import torch
import tqdm

# Importing necessary modules from aitviewer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node

# Importing local modules
from lib.pitch import SoccerPitch
from lib.utils import project_points

PITCH = SoccerPitch()
SOCCER_FIELD_LINES = PITCH.lines


def create_billboard(camera, img_folder, distance=200, draw_fn=None):
    """Create a billboard from a sequence of images."""
    img_paths = sorted(img_folder.glob("*.jpg"))
    H, W = camera.rows, camera.cols
    pc = Billboard.from_camera_and_distance(
        camera, distance, W, H, textures=[str(path) for path in img_paths], image_process_fn=draw_fn
    )
    return pc


def convert_video_to_images(video_path, output_folder):
    """Convert video to images."""
    output_folder.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Converting video to images:") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fp = output_folder / f"{frame_id:06d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), frame)
            frame_id += 1
            pbar.update(1)
    cap.release()


def create_smpl_sequences(params, names=None, colors=None, post_fk_func=None):
    """Create SMPL sequences for animation."""
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    num_subjects = len(params["global_orient"])

    names = names if names is not None else [f"SMPL_{i}" for i in range(num_subjects)]
    colors = colors if colors is not None else [(0.5, 0.5, 0.5, 1) for _ in range(num_subjects)]

    smpl_seqs = []
    for i in range(num_subjects):
        smpl_seq = SMPLSequence(
            poses_body=params["body_pose"][i],
            smpl_layer=smpl_layer,
            poses_root=params["global_orient"][i],
            betas=params["betas"][i],
            trans=params["transl"][i],
            is_rigged=False,
            post_fk_func=post_fk_func,
            name=names[i],
            color=colors[i],
        )
        smpl_seq.mesh_seq.compute_vertex_and_face_normals.cache_clear()
        smpl_seqs.append(smpl_seq)
    return smpl_seqs


def draw_field(frame, calib, img_shape):
    for line in SOCCER_FIELD_LINES:
        line = project_points(line, **calib, img_shape=img_shape)
        line = line.astype(np.int32)
        cv2.polylines(frame, [line], False, (0, 0, 255), 5)
    return frame


def make_draw_func(camera=None):
    def _draw_func(img, current_frame_id):
        if camera:
            current_frame_id = min(current_frame_id, len(camera["K"]) - 1)
            img_shape = img.shape
            calib = {
                "R": camera["R"][current_frame_id],
                "t": camera["t"][current_frame_id],
                "k": camera["k"][current_frame_id],
                "f": camera["K"][current_frame_id][0, 0],
                "principal_points": camera["K"][current_frame_id][:2, 2],
            }
            draw_field(img, calib, img_shape)
        return img

    return _draw_func


def make_post_fk_func(camera_params):
    R_all = torch.from_numpy(camera_params["R"]).float()
    t_all = torch.from_numpy(camera_params["t"]).float()
    k_all = torch.from_numpy(camera_params["k"]).float()

    def _post_fk_func(self, vertices, joints, current_frame_only):
        # apply rotation and translation
        nonlocal R_all, t_all, k_all
        R = R_all.to(vertices.device)
        t = t_all.to(vertices.device)
        k = k_all.to(vertices.device)
        if current_frame_only:
            R = R[self.current_frame_id][None]
            t = t[self.current_frame_id][None]
            k = k[self.current_frame_id][None]

        vertices_cam = (R[:, None] @ vertices[..., None]).squeeze(-1)
        vertices_cam += t[:, None]
        vertices_normalized = vertices_cam[..., :2] / vertices_cam[..., 2:]
        r = vertices_normalized.square().sum(-1, keepdims=True)
        # r.clamp_(0, 1)
        k1 = k[:, 0:1]
        k2 = k[:, 1:2]

        scale = 1 + k1[:, None] * r + k2[..., None] * r.square()
        vertices_cam[..., :2] *= scale

        # transform back to world coordinates
        vertices_cam -= t[:, None]
        vertices_cam = (R[:, None].transpose(-1, -2) @ vertices_cam[..., None]).squeeze(-1)
        vertices = vertices_cam
        return vertices, joints

    return _post_fk_func

def process_clip(clip_name):
    data_dir = Path("../data/WorldPoseDataset")
    smpl_param_path = data_dir / f"poses/{clip_name}.npz"
    video_path = data_dir / f"videos/{clip_name}.mp4"
    calibration_path = data_dir / f"cameras/{clip_name}.npz"
    if not smpl_param_path.exists():
        raise FileNotFoundError(f"SMPL parameters not found at {smpl_param_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration parameters not found at {calibration_path}")

    camera_params = dict(np.load(calibration_path))

    # Load SMPL parameters
    smpl_params = dict(np.load(smpl_param_path))
    colors = np.random.rand(len(smpl_params["betas"]), 4)
    colors[:, 3] = 1
    post_fk_func = make_post_fk_func(camera_params)
    smpl_seqs = create_smpl_sequences(smpl_params, colors=colors, post_fk_func=post_fk_func)

    # Setup camera and billboard
    camera = OpenCVCamera(camera_params["K"], camera_params["Rt"], 1920, 1080, viewer=viewer, name="Overlay")
    ## create a tmp folder and convert video to images
    img_folder = Path(f"outputs/{clip_name}")
    convert_video_to_images(video_path, img_folder)
    billboard = create_billboard(camera, img_folder, 200, make_draw_func(camera_params))
    # viewer.scene.add(billboard)
    # viewer.scene.add(camera)

    # # Add SMPL sequences to the scene
    # smpl_seq_node = Node(name="SMPL", n_frames=len(camera_params["K"]), is_selectable=False)
    # for seq in smpl_seqs:
    #     smpl_seq_node.add(seq)
    # viewer.scene.add(smpl_seq_node)

    # # Configure lighting
    # light = viewer.scene.lights[0]
    # light.shadow_enabled = True
    # light.azimuth = 270
    # light.elevation = 0
    # light.shadow_map_size = 64
    # light.shadow_map_near = 0.01
    # light.shadow_map_far = 50
    # viewer.scene.lights[1].shadow_enabled = False

    # # Finalize setup and run viewer
    # viewer.scene.floor.enabled = False
    # viewer.set_temp_camera(camera)
    # viewer.run()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="ARG_FRA_182345")
    args = parser.parse_args()

    # define constants
    C.z_up = True
    #clips = ['NET_ARG_001438', 'ARG_CRO_222132', 'FRA_MOR_232451', 'MOR_POR_183629', 'ENG_FRA_235059', 'ARG_FRA_203048', 'NET_ARG_222226', 'ARG_CRO_222717', 'ARG_CRO_222547', 'ARG_CRO_235121', 'CRO_MOR_183903', 'CRO_MOR_182607', 'MOR_POR_191519', 'ARG_FRA_201902', 'ENG_FRA_233519', 'NET_ARG_220032', 'BRA_KOR_221807', 'ARG_FRA_192649', 'CRO_MOR_181448', 'CRO_MOR_194948', 'NET_ARG_231259', 'CRO_MOR_182854', 'ARG_CRO_221657', 'MOR_POR_193355', 'MOR_POR_182352', 'BRA_KOR_232557', 'BRA_KOR_233015', 'FRA_MOR_220001', 'BRA_KOR_224459', 'ARG_CRO_234105', 'BRA_KOR_230549', 'MOR_POR_191625', 'CRO_MOR_193322', 'ENG_FRA_235745', 'ARG_FRA_193158', 'BRA_KOR_231503', 'MOR_POR_181952', 'CRO_MOR_184559', 'ARG_CRO_230507', 'ARG_CRO_232320', 'MOR_POR_184724', 'ENG_FRA_223446', 'ENG_FRA_232115', 'ARG_CRO_231510', 'CRO_MOR_182806', 'ARG_FRA_180702', 'ENG_FRA_223257', 'ENG_FRA_221238', 'FRA_MOR_222923', 'FRA_MOR_231753', 'BRA_KOR_222111', 'BRA_KOR_232126', 'NET_ARG_221729', 'ENG_FRA_232015', 'NET_ARG_004304', 'ENG_FRA_221122', 'CRO_MOR_194338', 'BRA_KOR_223206', 'ARG_CRO_220001', 'ENG_FRA_224248', 'MOR_POR_181500', 'ARG_CRO_223805', 'FRA_MOR_233043', 'NET_ARG_223806', 'CRO_MOR_192932', 'NET_ARG_222749', 'MOR_POR_192030', 'ENG_FRA_232558', 'ARG_FRA_200043', 'ARG_FRA_183429', 'ENG_FRA_224710', 'ENG_FRA_231054', 'BRA_KOR_224741', 'CRO_MOR_181141', 'NET_ARG_224119', 'BRA_KOR_222955', 'ARG_FRA_181108', 'ARG_CRO_220954', 'ARG_CRO_221101', 'ENG_FRA_232424', 'FRA_MOR_220401', 'CRO_MOR_184059', 'ARG_CRO_231839', 'ENG_FRA_233855', 'NET_ARG_231908', 'CRO_MOR_180400', 'NET_ARG_225042', 'BRA_KOR_221708', 'ARG_FRA_182345']
    clips = [s.replace(".npz", "") for s in os.listdir('../data/WorldPoseDataset/poses')]
    # Setup viewer and load data
    # viewer = Viewer(size=(1920, 1080))
    # clip_name = args.sequence
    for clip in clips:
        print(clip)
        #process_clip(clip)
    
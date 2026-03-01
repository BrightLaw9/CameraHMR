import os
from typing import Dict
from yacs.config import CfgNode as CN
curr_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(curr_dir, '../../')
DATASET_FOLDERS = {

    '3dpw-test-cam-smpl': os.path.join(base_dir, 'data/test-images/3DPW'),
    'coco-val-smpl': os.path.join(base_dir, 'data/test-images/COCO2017/images/'),
    'emdb-smpl': os.path.join(base_dir, 'data/test-images/EMDB'),
    'spec-test-smpl': os.path.join(base_dir, 'data/test-images/spec-syn'),
    'spec-mtp-test-smpl': os.path.join(base_dir, 'data/test-images/spec-mtp/'),
    'rich-smplx': os.path.join(base_dir, 'data/test-images/RICH'),
    'coco-val-smpl': os.path.join(base_dir, 'data/test-images/COCO2017/images'),

    'insta-1': os.path.join(base_dir, 'data/training-images/insta/images/'),
    'insta-2': os.path.join(base_dir, 'data/training-images/insta/images'),
    'aic': os.path.join(base_dir, 'data/training-images/aic/images'),
    'mpii-train':  os.path.join(base_dir, 'data/training-images/MPII-pose'),
    'coco-train':  os.path.join(base_dir, 'data/training-images/COCO'),

    'worldpose-train': os.path.join(base_dir, 'data/WorldPoseDataset/train/outputs/'),
    'worldpose-test': os.path.join(base_dir, 'data/WorldPoseDataset/val/outputs/'),

    #BEDLAM (SMPL)
    # 'agora-body-bbox44': os.path.join(base_dir, 'data/training-images/images'),
    # 'zoom-suburbd-bbox44': os.path.join(base_dir, 'data/training-images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png'),
    # 'closeup-suburba-bbox44': os.path.join(base_dir, 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    # 'closeup-suburbb-bbox44': os.path.join(base_dir, 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png'),
    # 'closeup-suburbc-bbox44': os.path.join(base_dir, 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png'),
    # 'closeup-suburbd-bbox44': os.path.join(base_dir, 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png'),
    # 'closeup-gym-bbox44': os.path.join(base_dir, 'data/training-images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png'),
    # 'zoom-gym-bbox44': os.path.join(base_dir, 'data/training-images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png'),
    # 'static-gym-bbox44': os.path.join(base_dir, 'data/training-images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png'),
    # 'static-office-bbox44': os.path.join(base_dir, 'data/training-images/20221013_3_250_batch01hand_static_bigOffice_6fps/png'),
    # 'orbit-office-bbox44': os.path.join(base_dir, 'data/training-images/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png'),
    # 'orbit-archviz-15-bbox44': os.path.join(base_dir, 'data/training-images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    # 'orbit-archviz-19-bbox44': os.path.join(base_dir, 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png'),
    # 'orbit-archviz-12-bbox44': os.path.join(base_dir, 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png'),
    # 'orbit-archviz-10-bbox44': os.path.join(base_dir, 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png'),
    # 'static-hdri-bbox44': os.path.join(base_dir, 'data/training-images/20221010_3_1000_batch01hand_6fps/png'),
    # 'static-hdri-zoomed-bbox44': os.path.join(base_dir, 'data/training-images/20221017_3_1000_batch01hand_6fps/png'),
    # 'staticzoomed-suburba-frameocc-bbox44': os.path.join(base_dir, 'data/training-images/20221017_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    # 'zoom-suburbb-frameocc-bbox44': os.path.join(base_dir, 'data/training-images/20221018_1_250_batch01hand_zoom_suburb_b_6fps/png'),
    # 'static-hdri-frameocc-bbox44': os.path.join(base_dir, 'data/training-images/20221018_3-8_250_batch01hand_6fps/png'),
    # 'orbit-archviz-objocc-bbox44': os.path.join(base_dir, 'data/training-images/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    # 'pitchup-stadium-bbox44': os.path.join(base_dir, 'data/training-images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png'),
    # 'pitchdown-stadium-bbox44': os.path.join(base_dir, 'data/training-images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png'),
    # 'static-hdri-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_3_250_highbmihand_6fps/png'),
    # 'closeup-suburbb-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png'),
    # 'closeup-suburbc-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png'),
    # 'static-stadium-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_3-8_250_highbmihand_static_stadium_6fps/png'),
    # 'orbit-stadium-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_3-8_250_highbmihand_orbit_stadium_6fps/png'),
    # 'static-suburbd-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png'),
    # 'zoom-gym-bmi-bbox44': os.path.join(base_dir, 'data/training-images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png'),
    # 'static-office-hair-bbox44': os.path.join(base_dir, 'data/training-images/20221022_3_250_batch01handhair_static_bigOffice_30fps/png'),
    # 'zoom-suburbd-hair-bbox44': os.path.join(base_dir, 'data/training-images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png'),
    # 'static-gym-hair-bbox44': os.path.join(base_dir, 'data/training-images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png'),

    
    # BEDLAM 1 (SMPLX)
    # 'agora-body-bbox44-smplx':  'data/training-images/agora/images',
    # 'zoom-suburbd-bbox44-smplx':  'data/training-images//20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png',
    # 'closeup-suburba-bbox44-smplx': 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png',
    # 'closeup-suburbb-bbox44-smplx': 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png',
    # 'closeup-suburbc-bbox44-smplx': 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png',
    # 'closeup-suburbd-bbox44-smplx': 'data/training-images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png',
    # 'closeup-gym-bbox44-smplx': 'data/training-images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png',
    # 'zoom-gym-bbox44-smplx': 'data/training-images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png',
    # 'static-gym-bbox44-smplx': 'data/training-images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png',
    # 'static-office-bbox44-smplx': 'data/training-images/20221013_3_250_batch01hand_static_bigOffice_6fps/png',
    # 'orbit-office-bbox44-smplx': 'data/training-images/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png',
    # 'orbit-archviz-15-bbox44-smplx': 'data/training-images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png',
    # 'orbit-archviz-19-bbox44-smplx': 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png',
    # 'orbit-archviz-12-bbox44-smplx': 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png',
    # 'orbit-archviz-10-bbox44-smplx': 'data/training-images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png',
    # 'static-hdri-bbox44-smplx': 'data/training-images/20221010_3_1000_batch01hand_6fps/png',
    # 'static-hdri-zoomed-bbox44-smplx': 'data/training-images/20221017_3_1000_batch01hand_6fps/png',
    # 'staticzoomed-suburba-frameocc-bbox44-smplx': 'data/training-images/20221017_1_250_batch01hand_closeup_suburb_a_6fps/png',
    # 'zoom-suburbb-frameocc-bbox44-smplx': 'data/training-images/20221018_1_250_batch01hand_zoom_suburb_b_6fps/png',
    # 'static-hdri-frameocc-bbox44-smplx': 'data/training-images/20221018_3-8_250_batch01hand_6fps/png',
    # 'orbit-archviz-objocc-bbox44-smplx': 'data/training-images/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png',
    # 'pitchup-stadium-bbox44-smplx': 'data/training-images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png',
    # 'pitchdown-stadium-bbox44-smplx': 'data/training-images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png',
    # 'static-hdri-bmi-bbox44-smplx': 'data/training-images/20221019_3_250_highbmihand_6fps/png',
    # 'closeup-suburbb-bmi-bbox44-smplx': 'data/training-images/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png',
    # 'closeup-suburbc-bmi-bbox44-smplx': 'data/training-images/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png',
    # 'static-stadium-bmi-bbox44-smplx': 'data/training-images/20221019_3-8_250_highbmihand_static_stadium_6fps/png',
    # 'orbit-stadium-bmi-bbox44-smplx': 'data/training-images/20221019_3-8_250_highbmihand_orbit_stadium_6fps/png',
    # 'static-suburbd-bmi-bbox44-smplx': 'data/training-images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png',
    # 'zoom-gym-bmi-bbox44-smplx': 'data/training-images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png',
    # 'static-office-hair-bbox44-smplx': 'data/training-images/20221022_3_250_batch01handhair_static_bigOffice_30fps/png',
    # 'zoom-suburbd-hair-bbox44-smplx': 'data/training-images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png',
    # 'static-gym-hair-bbox44-smplx': 'data/training-images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png',

    # BEDLAM 2
    # "city-dolly-moyo1-smplx-notest": "data/training-images/bedlam_v2/20240425_1_171_citysample_dolly/png",
    # "yoga-orbit-moyo-smplx-notest": "data/training-images/bedlam_v2/20240416_1_171_yogastudio_orbit_timeofday/png",
    # "yoga-static-moyo-smplx-notest": "data/training-images/bedlam_v2/20240423_1_171_yogastudio_staticloc_timeofday/png",
    # "city-orbit-moyo1-smplx-notest": "data/training-images/bedlam_v2/20240424_1_171_citysample_orbit/png",
    # "hdri-moyo-smplx-notest": "data/training-images/bedlam_v2/20240425_1_171_hdri/png",
    # "city-orbit-moyo2-smplx-notest": "data/training-images/bedlam_v2/20240426_5_100_citysample_orbit/png",
    # "stadium-moyo-smplx-notest": "data/training-images/bedlam_v2/20240429_1_171_stadium/png",
    # "city-dolly-moyo2-smplx-notest": "data/training-images/bedlam_v2/20240502_5_200_citysample_dolly/png",
    # "hdri-moyo2-smplx-notest": "data/training-images/bedlam_v2/20240506_10_200_hdri/png",
    # "city-orbit-moyo3-smplx-notest": "data/training-images/bedlam_v2/20240506_5_200_citysample_orbit/png",
    # "city-dollyz-moyo-smplx-notest": "data/training-images/bedlam_v2/20240507_5_200_citysample_dollyz/png",
    # "city-tracking-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240514_1_1001_citysample_tracking/png",
    # "city-tracking-b2v02-smplx-notest": "data/training-images/bedlam_v2/20240604_5_500_citysample_tracking/png",
    # "bus-tracking-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240605_3_500_busstation_tracking/png",
    # "bus-orbit-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240606_4_250_busstation_orbit/png",
    # "stadium-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240606_1_500_stadium_closeup/png",
    # "archmodel-dolly-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240611_5_250_archmodelsvol8_dolly/png",
    # "citynight-dolly-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240611_5_200_citysamplenight_dolly/png",
    # "hdri-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240613_1_200_hdri/png",
    # "citynight-tracking-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240614_5_200_citysamplenight_tracking/png",
    # "hdri-b2v02-smplx-notest": "data/training-images/bedlam_v2/20240614_1_300_hdri/png",
    # "hdri-b2v03-smplx-notest": "data/training-images/bedlam_v2/20240617_10_500_hdri/png",
    # "ai0805-orbit-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240618_1_500_ai0805_orbit/png",
    # "ai1004-orbit-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240619_2_250_ai1004_orbit/png",
    # "ai1004-tracking-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240619_1_250_ai1004_tracking/png",
    # "archmodel-dollyz-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240620_5_250_archmodelsvol8_dollyz/png",
    # "archmodel-tracking-b2v01-smplx-notest": "data/training-images/bedlam_v2/20240621_1_250_archmodelsvol8_tracking/png",
    # "hdri-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240625_1_2337_hdri/png",
    # "ai1004-tracking-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240628_1_250_ai1004_tracking/png",
    # "bus-tracking-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240628_4_250_busstation_orbit/png",
    # "ai0901-lookat-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240701_1_250_ai0901_lookat/png",
    # "ai0901-orbit-portrait-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240703_1_250_ai0901_orbit_portrait/png",
    # "ai0901-static-portrait-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240708_1_250_ai0901_static_portrait/png",
    # "archmodel-zoom-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240709_5_250_archmodelsvol8_zoom/png",
    # "ai0805-orbit-portrait-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240710_1_250_ai0805_orbit_portrait/png",
    # "bus-orbit-zoom-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240711_5-10_250_busstation_orbit_zoom/png",
    # "ai0805-vcam-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240725_1_250_ai0805_vcam/png",
    # "ai0805-vcam-b2v12-smplx-notest": "data/training-images/bedlam_v2/20240726_1_250_ai0805_vcam/png",
    # "ai1004-vcam-portrait-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240729_1_250_ai1004_vcam/png",
    # "ai1101-vcam-portrait-b2v11-smplx-notest": "data/training-images/bedlam_v2/20240730_1_250_ai1101_vcam/png",
    # "hdri-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240731_1_1827_hdri/png",
    # "bus-orbit-zoom-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240805_5-10_250_busstation_orbit_zoom/png",
    # "ai1101-vcam-portrait-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240806_1_250_ai1101_vcam/png",
    # "ai1105-vcam-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240808_1_250_ai1105_vcam/png",
    # "ai1102-vcam-portrait-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240809_1_250_ai1102_vcam/png",
    # "ai1004-tracking-b2v21-smplx-notest": "data/training-images/bedlam_v2/20240813_1_250_ai1004_tracking/png",
  
  
    # "bus-orbit-zoom-b2v22-smplx-notest": "data/training-images/bedlam_v2/20241001_5-10_250_busstation_orbit_zoom/png",
    # "archmodel-tracking-b2v02-smplx-notest": "data/training-images/bedlam_v2/20241107_1_250_archmodelsvol8_tracking/png",
    # "hdri-b2v30-smplx-notest": "data/training-images/bedlam_v2/20241114_1_4619_hdri/png",
    # "hdri-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241204_1_2120_hdri/png",
    # "rome-dollyz-zoom-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241210_5-10_250_rome_dollyz_zoom/png",
    # "rome-orbit-zoom-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241211_5-10_250_rome_orbit_zoom/png",
    # "rome-dolly-zoom-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241212_5-10_250_rome_dolly_zoom/png",
    # "rome-tracking-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241213_1_250_rome_tracking/png",
    # "rome-vcam-portrait-b2v40-smplx-notest": "data/training-images/bedlam_v2/20241217_1_250_rome_vcam/png",
    # "chemicalplant-dollyz-zoom-b2v30-smplx-notest": "data/training-images/bedlam_v2/20241219_5_250_chemicalplant_dollyz_zoom/png",
    # "rome-vcam-portrait-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250103_1_250_rome_vcam/png",
    # "chemicalplant-vcam-portrait-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250110_1_250_chemicalplant_vcam/png",
    # "rome-vcam-b2v31-smplx-notest": "data/training-images/bedlam_v2/20250113_1_250_rome_vcam/png",
    # "chemicalplant-dolly-zoom-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250114_4-5_250_chemicalplant_dolly_zoom/png",
    # "chemicalplant-vcamego-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250123_1_250_chemicalplant_vcamego/png",
    # "ai1102-vcamego-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250131_1_250_ai1102_vcamego/png",
    # "yakohama-vcamego-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250206_4-7_250_yakohama_vcamego_approach/png",
    # "ai1105-upperbody-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250211_1_250_ai1105_upperbody/png",
    # "yakohama-upperbody-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250212_1_250_yakohama_upperbody/png",
    # "chemicalplant-upperbody-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250214_1_250_chemicalplant_upperbody/png",
    # "middleeasy-upperbody-b2v30-smplx-notest": "data/training-images/bedlam_v2/20250218_2-3_250_middleeast_upperbody/png",
    # "middleeast-vacam-b2v40-smplx-notest": "data/training-images/bedlam_v2/20250219_3-4_250_middleeast_vcam_approach/png",

}

DATASET_FILES = [
    {
        '3dpw-test-cam-smpl': os.path.join(base_dir, 'data/test-labels/3dpw_test.npz'),
        'emdb-smpl': os.path.join(base_dir, 'data/test-labels/emdb_test.npz'),
        'rich-smplx': os.path.join(base_dir, 'data/test-labels/rich_test.npz'),
        'spec-test-smpl': os.path.join(base_dir, 'data/test-labels/spec_test.npz'),
        'spec-mtp-test-smpl': os.path.join(base_dir, 'data/test-labels/spec_mtp_test.npz'),
        'coco-val-smpl': os.path.join(base_dir, 'data/test-labels/coco_val.npz'),
        'worldpose-test': os.path.join(base_dir, 'data/WorldPoseDataset/val/'),
    },
    {
        'aic': os.path.join(base_dir, 'data//training-labels/aic-release.npz'),
        'insta-1': os.path.join(base_dir, 'data//training-labels/insta1-release.npz'),
        'insta-2': os.path.join(base_dir, 'data//training-labels/insta2-release.npz'),
        'coco-train': os.path.join(base_dir, 'data/training-labels/coco-release.npz'),
        'mpii-train': os.path.join(base_dir, 'data/training-labels/mpii-release.npz'),
        
        # Only provide the directory to load all the npz files
        'worldpose-train': os.path.join(base_dir, 'data/WorldPoseDataset/train/'),
        # CameraHMR
        # BEDLAM1 SMPL
        # 'agora-body-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/agora.npz'),
        # 'zoom-suburbd-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz'),
        # 'closeup-suburba-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        # 'closeup-suburbb-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_b_6fps.npz'),
        # 'closeup-suburbc-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_c_6fps.npz'),
        # 'closeup-suburbd-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz'),
        # 'closeup-gym-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.npz'),
        # 'zoom-gym-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz'),
        # 'static-gym-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz'),
        # 'static-office-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221013_3_250_batch01hand_static_bigOffice_6fps.npz'),
        # 'orbit-office-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221013_3_250_batch01hand_orbit_bigOffice_6fps.npz'),
        # 'orbit-archviz-15-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        # 'orbit-archviz-19-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.npz'),
        # 'orbit-archviz-12-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.npz'),
        # 'orbit-archviz-10-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.npz'),
        # 'static-hdri-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221010_3_1000_batch01hand_6fps.npz'),
        # 'static-hdri-zoomed-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221017_3_1000_batch01hand_6fps.npz'),
        # 'staticzoomed-suburba-frameocc-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221017_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        # 'pitchup-stadium-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.npz'),
        # 'static-hdri-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221019_3_250_highbmihand_6fps.npz'),
        # 'closeup-suburbb-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221019_1_250_highbmihand_closeup_suburb_b_6fps.npz'),
        # 'closeup-suburbc-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221019_1_250_highbmihand_closeup_suburb_c_6fps.npz'),
        # 'static-suburbd-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221019_3-8_1000_highbmihand_static_suburb_d_6fps.npz'),
        # 'zoom-gym-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221020_3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.npz'),
        # 'pitchdown-stadium-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.npz'),
        # 'static-office-hair-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221022_3_250_batch01handhair_static_bigOffice_30fps.npz'),
        # 'zoom-suburbd-hair-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221024_10_100_batch01handhair_zoom_suburb_d_30fps.npz'),
        # 'static-gym-hair-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz'),
        # 'orbit-stadium-bmi-bbox44': os.path.join(base_dir, 'data/training-labels/bedlam-labels/20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz'),

        # # BEDLAM1 SMPLX
        # 'agora-bfh-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/agora.npz',
        # 'agora-body-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/agora.npz',
        # 'zoom-suburbd-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221010_3-10_500_batch01hand_zoom_suburb_d.npz',
        # 'closeup-suburba-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221011_1_250_batch01hand_closeup_suburb_a.npz',
        # 'closeup-suburbb-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221011_1_250_batch01hand_closeup_suburb_b.npz',
        # 'closeup-suburbc-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221011_1_250_batch01hand_closeup_suburb_c.npz',
        # 'closeup-suburbd-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221011_1_250_batch01hand_closeup_suburb_d.npz',
        # 'closeup-gym-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221012_1_500_batch01hand_closeup_highSchoolGym.npz',
        # 'zoom-gym-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221012_3-10_500_batch01hand_zoom_highSchoolGym.npz',
        # 'static-gym-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221013_3-10_500_batch01hand_static_highSchoolGym.npz',
        # 'static-office-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221013_3_250_batch01hand_static_bigOffice.npz',
        # 'orbit-office-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221013_3_250_batch01hand_orbit_bigOffice.npz',
        # 'orbit-archviz-15-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221014_3_250_batch01hand_orbit_archVizUI3_time15.npz',
        # 'orbit-archviz-19-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221015_3_250_batch01hand_orbit_archVizUI3_time19.npz',
        # 'orbit-archviz-12-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221015_3_250_batch01hand_orbit_archVizUI3_time12.npz',
        # 'orbit-archviz-10-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221015_3_250_batch01hand_orbit_archVizUI3_time10.npz',
        # 'static-hdri-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221010_3_1000_batch01hand.npz',
        # 'static-hdri-zoomed-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221017_3_1000_batch01hand.npz',
        # 'staticzoomed-suburba-frameocc-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221017_1_250_batch01hand_closeup_suburb_a.npz',
        # 'pitchup-stadium-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221018_3-8_250_batch01hand_pitchUp52_stadium.npz',
        # 'static-hdri-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221019_3_250_highbmihand.npz',
        # 'closeup-suburbb-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221019_1_250_highbmihand_closeup_suburb_b.npz',
        # 'closeup-suburbc-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221019_1_250_highbmihand_closeup_suburb_c.npz',
        # 'static-suburbd-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221019_3-8_1000_highbmihand_static_suburb_d.npz',
        # 'zoom-gym-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221020_3-8_250_highbmihand_zoom_highSchoolGym_a.npz',
        # 'pitchdown-stadium-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221018_3-8_250_batch01hand_pitchDown52_stadium.npz',
        # 'static-office-hair-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221022_3_250_batch01handhair_static_bigOffice.npz',
        # 'zoom-suburbd-hair-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221024_10_100_batch01handhair_zoom_suburb_d.npz',
        # 'static-gym-hair-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221024_3-10_100_batch01handhair_static_highSchoolGym.npz',
        # 'orbit-stadium-bmi-bbox44-smplx': 'data/training-labels/bedlam-labels-v2-format/20221019_3-8_250_highbmihand_orbit_stadium.npz',

        
        # # BEDLAM2

        # "city-dolly-moyo1-smplx-notest": "data/training-labels/bedlam_v2/20240425_1_171_citysample_dolly.npz",
        # "yoga-orbit-moyo-smplx-notest": "data/training-labels/bedlam_v2/20240416_1_171_yogastudio_orbit_timeofday.npz",
        # "yoga-static-moyo-smplx-notest": "data/training-labels/bedlam_v2/20240423_1_171_yogastudio_staticloc_timeofday.npz",
        # "city-orbit-moyo1-smplx-notest": "data/training-labels/bedlam_v2/20240424_1_171_citysample_orbit.npz",
        # "hdri-moyo-smplx-notest": "data/training-labels/bedlam_v2/20240425_1_171_hdri.npz",
        # "city-orbit-moyo2-smplx-notest": "data/training-labels/bedlam_v2/20240426_5_100_citysample_orbit.npz",
        # "stadium-moyo-smplx-notest": "data/training-labels/bedlam_v2/20240429_1_171_stadium.npz",
        # "city-dolly-moyo2-smplx-notest": "data/training-labels/bedlam_v2/20240502_5_200_citysample_dolly.npz",
        # "hdri-moyo2-smplx-notest": "data/training-labels/bedlam_v2/20240506_10_200_hdri.npz",
        # "city-orbit-moyo3-smplx-notest": "data/training-labels/bedlam_v2/20240506_5_200_citysample_orbit.npz",
        # "city-dollyz-moyo-smplx-notest": "data/training-labels/bedlam_v2/20240507_5_200_citysample_dollyz.npz",
        # "city-tracking-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240514_1_1001_citysample_tracking.npz",
        # "city-tracking-b2v02-smplx-notest": "data/training-labels/bedlam_v2/20240604_5_500_citysample_tracking.npz",
        # "bus-tracking-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240605_3_500_busstation_tracking.npz",
        # "bus-orbit-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240606_4_250_busstation_orbit.npz",
        # "stadium-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240606_1_500_stadium_closeup.npz",
        # "archmodel-dolly-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240611_5_250_archmodelsvol8_dolly.npz",
        # "citynight-dolly-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240611_5_200_citysamplenight_dolly.npz",
        # "hdri-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240613_1_200_hdri.npz",
        # "citynight-tracking-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240614_5_200_citysamplenight_tracking.npz",
        # "hdri-b2v02-smplx-notest": "data/training-labels/bedlam_v2/20240614_1_300_hdri.npz",
        # "hdri-b2v03-smplx-notest": "data/training-labels/bedlam_v2/20240617_10_500_hdri.npz",
        # "ai0805-orbit-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240618_1_500_ai0805_orbit.npz",
        # "ai1004-orbit-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240619_2_250_ai1004_orbit.npz",
        # "ai1004-tracking-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240619_1_250_ai1004_tracking.npz",
        # "archmodel-dollyz-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240620_5_250_archmodelsvol8_dollyz.npz",
        # "archmodel-tracking-b2v01-smplx-notest": "data/training-labels/bedlam_v2/20240621_1_250_archmodelsvol8_tracking.npz",
        # "hdri-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240625_1_2337_hdri.npz",
        # "ai1004-tracking-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240628_1_250_ai1004_tracking.npz",
        # "bus-tracking-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240628_4_250_busstation_orbit.npz",
        # "ai0901-lookat-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240701_1_250_ai0901_lookat.npz",
        # "ai0901-orbit-portrait-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240703_1_250_ai0901_orbit_portrait.npz",
        # "ai0901-static-portrait-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240708_1_250_ai0901_static_portrait.npz",
        # "archmodel-zoom-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240709_5_250_archmodelsvol8_zoom.npz",
        # "ai0805-orbit-portrait-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240710_1_250_ai0805_orbit_portrait.npz",
        # "bus-orbit-zoom-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240711_5-10_250_busstation_orbit_zoom.npz",
        # "ai0805-vcam-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240725_1_250_ai0805_vcam.npz",
        # "ai0805-vcam-b2v12-smplx-notest": "data/training-labels/bedlam_v2/20240726_1_250_ai0805_vcam.npz",
        # "ai1004-vcam-portrait-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240729_1_250_ai1004_vcam.npz",
        # "ai1101-vcam-portrait-b2v11-smplx-notest": "data/training-labels/bedlam_v2/20240730_1_250_ai1101_vcam.npz",
        # "hdri-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240731_1_1827_hdri.npz",
        # "bus-orbit-zoom-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240805_5-10_250_busstation_orbit_zoom.npz",
        # "ai1101-vcam-portrait-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240806_1_250_ai1101_vcam.npz",
        # "ai1105-vcam-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240808_1_250_ai1105_vcam.npz",
        # "ai1102-vcam-portrait-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240809_1_250_ai1102_vcam.npz",
        # "ai1004-tracking-b2v21-smplx-notest": "data/training-labels/bedlam_v2/20240813_1_250_ai1004_tracking.npz",

        # "bus-orbit-zoom-b2v22-smplx-notest": "data/training-labels/bedlam_v2/20241001_5-10_250_busstation_orbit_zoom.npz",
        # "archmodel-tracking-b2v02-smplx-notest": "data/training-labels/bedlam_v2/20241107_1_250_archmodelsvol8_tracking.npz",
        # "hdri-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20241114_1_4619_hdri.npz",
        # "hdri-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241204_1_2120_hdri.npz",
        # "rome-dollyz-zoom-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241210_5-10_250_rome_dollyz_zoom.npz",
        # "rome-orbit-zoom-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241211_5-10_250_rome_orbit_zoom.npz",
        # "rome-dolly-zoom-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241212_5-10_250_rome_dolly_zoom.npz",
        # "rome-tracking-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241213_1_250_rome_tracking.npz",
        # "rome-vcam-portrait-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20241217_1_250_rome_vcam.npz",
        # "chemicalplant-dollyz-zoom-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20241219_5_250_chemicalplant_dollyz_zoom.npz",
        # "rome-vcam-portrait-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250103_1_250_rome_vcam.npz",
        # "chemicalplant-vcam-portrait-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250110_1_250_chemicalplant_vcam.npz",
        # "rome-vcam-b2v31-smplx-notest": "data/training-labels/bedlam_v2/20250113_1_250_rome_vcam.npz",
        # "chemicalplant-dolly-zoom-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250114_4-5_250_chemicalplant_dolly_zoom.npz",
        # "chemicalplant-vcamego-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250123_1_250_chemicalplant_vcamego.npz",
        # "ai1102-vcamego-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250131_1_250_ai1102_vcamego.npz",
        # "yakohama-vcamego-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250206_4-7_250_yakohama_vcamego_approach.npz",
        # "ai1105-upperbody-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250211_1_250_ai1105_upperbody.npz",
        # "yakohama-upperbody-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250212_1_250_yakohama_upperbody.npz",
        # "chemicalplant-upperbody-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250214_1_250_chemicalplant_upperbody.npz",
        # "middleeasy-upperbody-b2v30-smplx-notest": "data/training-labels/bedlam_v2/20250218_2-3_250_middleeast_upperbody.npz",
        # "middleeast-vacam-b2v40-smplx-notest": "data/training-labels/bedlam_v2/20250219_3-4_250_middleeast_vcam_approach.npz",

    }
]

def to_lower(x: Dict) -> Dict:
    return {k.lower(): v for k, v in x.items()}
_C = CN(new_allowed=True)

_C.GENERAL = CN(new_allowed=True)
_C.GENERAL.RESUME = True
_C.GENERAL.TIME_TO_RUN = 3300
_C.GENERAL.VAL_STEPS = 100
_C.GENERAL.LOG_STEPS = 100
_C.GENERAL.CHECKPOINT_STEPS = 20000
_C.GENERAL.CHECKPOINT_DIR = "checkpoints"
_C.GENERAL.SUMMARY_DIR = "tensorboard"
_C.GENERAL.NUM_GPUS = 1
_C.GENERAL.NUM_WORKERS = 4
_C.GENERAL.MIXED_PRECISION = True
_C.GENERAL.ALLOW_CUDA = True
_C.GENERAL.PIN_MEMORY = False
_C.GENERAL.DISTRIBUTED = False
_C.GENERAL.LOCAL_RANK = 0
_C.GENERAL.USE_SYNCBN = False
_C.GENERAL.WORLD_SIZE = 1

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False
_C.TRAIN.NORMALIZE_PER_IMAGE = False
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.CLIP_GRAD_VALUE = 1.0
_C.LOSS_WEIGHTS = CN(new_allowed=True)

_C.DATASETS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
_C.MODEL.IMAGE_SIZE = 224

_C.EXTRA = CN(new_allowed=True)
_C.EXTRA.FOCAL_LENGTH = 5000

_C.DATASETS.CONFIG = CN(new_allowed=True)
_C.DATASETS.CONFIG.SCALE_FACTOR = 0.3
_C.DATASETS.CONFIG.ROT_FACTOR = 30
_C.DATASETS.CONFIG.TRANS_FACTOR = 0.02
_C.DATASETS.CONFIG.COLOR_SCALE = 0.2
_C.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
_C.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
_C.DATASETS.CONFIG.DO_FLIP = True
_C.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
_C.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10
_C.DATASETS.CONFIG.USE_ALB = True
_C.DATASETS.CONFIG.ALB_PROB = 0.3

def default_config() -> CN:
    return _C.clone()

def dataset_config() -> CN:
    cfg = CN(new_allowed=True)
    cfg.freeze()
    return cfg



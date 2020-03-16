from easydict import EasyDict as edict
import numpy as np
import os
YoloConfig=edict()
YoloConfig.anchors=np.array([
    [4.9434993, 1.516986],
    [2.1259836, 1.6779645],
    [19.452609, 17.815241],
    [3.1458852, 2.4994355],
    [15.0302664, 2.3736405],
    [1.2374577, 2.8255595],
    [5.5330938, 3.605915],
    [2.4232311, 0.8086055],
    [0.3672315, 0.6450615],
    [1.3549788, 1.2046775],
    [0.9085392, 0.726555],
    [0.772209, 2.031382],
    [4.0958478, 9.108235],
    [0.5070438, 1.26041],
    [10.0207692, 6.877788],
    [1.9708173, 4.677844]
])
YoloConfig.num_anchors=YoloConfig.anchors.shape[0]
YoloConfig.classes=["Vehicle","Bicycle","Pedestrian","Unknown_unmovable"]
YoloConfig.num_classes=len(YoloConfig.classes)
YoloConfig.seg_thresh=0.5
YoloConfig.det_thresh=0.6
YoloConfig.nms_thresh=0.45

rootDir = "~/dataset/yolo_camera_detector/yolo3d_1128"
YoloConfig.model=edict()
YoloConfig.model.deploy_path=os.path.join(rootDir,"deploy.pt")
YoloConfig.model.weight_path=os.path.join(rootDir,"deploy.md")

YoloConfig.input_width=960
YoloConfig.input_height=384

YoloConfig.resize_or_crop="resize"

CameraConfig=edict()
CameraConfig.intrinsic_mat=np.array([
    [2012.47, 0.0, 988],
    [0.0, 2011.89, 659.86],
    [0.0, 0.0, 1.0]
])
CameraConfig.w=1920
CameraConfig.h=1080
CameraConfig.distort_params=np.array([-0.50578, 0.10144, -0.00745, 0.00664, 0.0])

CameraConfig.kitti=edict()
CameraConfig.kitti.cx=6.095593000000e+02
CameraConfig.kitti.cy=1.728540000000e+02
CameraConfig.kitti.fx=7.215377000000e+02
CameraConfig.kitti.fy=7.215377000000e+02
CameraConfig.kitti.w=1242
CameraConfig.kitti.h=375
CameraConfig.kitti.intrinsic_mat=np.array([
    [CameraConfig.kitti.fx, 0.0, CameraConfig.kitti.cx],
    [0.0, CameraConfig.kitti.fy, CameraConfig.kitti.cy],
    [0.0, 0.0, 1.0]
])
CameraConfig.kitti.distort_params=np.array([0.0,0.0,0.0,0.0,0.0])


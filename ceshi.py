#from __future__ import division
from yolo3d.config import YoloConfig,CameraConfig
from yolo3d.yolo3d import obj_3d,Yolo3D
from yolo3d.show_fun import *
from D2toD3.camera_converter import GeometryCameraConverter
import numpy as np
import cv2

im=cv2.imread("000002.png")
detector=Yolo3D(YoloConfig)
gcc=GeometryCameraConverter()
gcc.init_camera_model(CameraConfig.kitti.intrinsic_mat,CameraConfig.kitti.w,CameraConfig.kitti.h,CameraConfig.kitti.distort_params)
detector.init_net()
detector.init_params()
original_im=im.copy()
objs=detector.process_img(im)
gcc.convert(objs)
draw_25d_box(original_im,objs)
print("obj 1 distance:{}".format(objs[1].distance))
print("obj 1 center:{}".format(objs[1].center))
#draw_2d_box(original_im,objs)
#draw_3d_box(original_im,objs)
#draw_2d_box(original_im,objs)
#draw_3dim(original_im,objs)
#cv2.imshow("im",original_im)
#cv2.waitKey(0)


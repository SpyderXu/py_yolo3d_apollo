#from __future__ import division
from yolo3d.config import YoloConfig,CameraConfig
from yolo3d.yolo3d import obj_3d,Yolo3D
from yolo3d.show_fun import *
from D2toD3.camera_converter import GeometryCameraConverter
import numpy as np
import cv2
import os

detector=Yolo3D(YoloConfig)
gcc=GeometryCameraConverter()
gcc.init_camera_model(CameraConfig.intrinsic_mat,CameraConfig.w,CameraConfig.h,CameraConfig.distort_params)
detector.init_net()
detector.init_params()
#rootDir="/home/cvpr/dataset/KITTI/Tracking/training/image_02/0011"
rootDir="/home/cvpr/dataset/baidu/trajectory/asdt_sample_image/sample_image_7"
filenames=os.listdir(rootDir)
filenames.sort(key=str.lower)
for filename in filenames:
    #background = cv2.imread("back.jpeg")
    #draw_axis(background)
    filepath=os.path.join(rootDir,filename)
    im = cv2.imread(filepath)
    #im=im[:375,141:141+960,:]

    original_im = im.copy()
    objs = detector.process_img(im)
    gcc.convert(objs)
    #draw_25d_box(original_im, objs)
    #show_fov(background,objs)
    #draw_2d_box(original_im,objs)
    draw_3d_box(original_im,objs)
    # draw_2d_box(original_im,objs)
    # draw_3dim(original_im,objs)
    #background=cv2.resize(background,(500,500))
    cv2.imshow("im", original_im)
    #cv2.imshow("fov",background)
    k=cv2.waitKey(2)
    if k == ord('q'):
        break



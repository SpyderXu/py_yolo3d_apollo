import numpy as np
from yolo3d.config import CameraConfig
from D2toD3.camera_converter import GeometryCameraConverter
gcc=GeometryCameraConverter()
gcc.init_camera_model(CameraConfig.kitti.intrinsic_mat,CameraConfig.kitti.w,CameraConfig.kitti.h,CameraConfig.kitti.distort_params)
deg_alpha=1.5
w=3
h=2
l=5
h_half=h/2.0
w_half=w/2.0
l_half=l/2.0
upper_left=[10,10]
lower_right=[100,200]


corners = np.zeros((8, 3))
corners[0] = np.array([l_half, h_half, w_half])
corners[1] = np.array([l_half, h_half, -1 * w_half])
corners[2] = np.array([-1 * l_half, h_half, -1 * w_half])
corners[3] = np.array([-1 * l_half, h_half, w_half])
corners[4] = np.array([l_half, -1 * h_half, w_half])
corners[5] = np.array([l_half, -1 * h_half, -1 * w_half])
corners[6] = np.array([-1 * l_half, -1 * h_half, -1 * w_half])
corners[7] = np.array([-1 * l_half, -1 * h_half, w_half])

corners=gcc.rotate(deg_alpha,corners)

middle_v=np.array([0.0,0.0,20.0])
center_pixel=gcc.camera_model.project(middle_v)


max_pixel_x=float("-inf")
min_pixel_x=float("inf")
max_pixel_y=float("-inf")
min_pixel_y=float("inf")

for i in range(0,corners.shape[0]):
    point_2d=gcc.camera_model.project(corners[i]+middle_v)
    min_pixel_x=min(min_pixel_x,point_2d[0])
    max_pixel_x=max(max_pixel_x,point_2d[0])
    min_pixel_y=min(min_pixel_y,point_2d[1])
    max_pixel_y=max(max_pixel_y,point_2d[1])

relative_x=(center_pixel[0]-min_pixel_x)/(max_pixel_x-min_pixel_x)
relative_y=(center_pixel[1]-min_pixel_y)/(max_pixel_y-min_pixel_y)

mass_center_pixel=np.zeros(2)
mass_center_pixel[0]=(lower_right[0]-upper_left[0])*relative_x+upper_left[0]
mass_center_pixel[1]=(lower_right[1]-upper_left[1])*relative_y+upper_left[1]
print("mass center pixel:",mass_center_pixel)
mass_center_v=gcc.camera_model.unproject(mass_center_pixel)
mass_center_v=gcc.MakeUnit(mass_center_v)

for distance in range(10,150,10):
    distance_mass_v=mass_center_v*distance
    test_pixel=gcc.camera_model.project(distance_mass_v)
    print("distance:{}".format(distance))
    print("test_pixel:",test_pixel)
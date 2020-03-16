from __future__ import division
from camera import CameraDistort
import numpy as np

camera_model=CameraDistort()
w=1920
h=1080
intrinsic_mat=np.array([
    [2012.47, 0.0, 988],
    [0.0, 2011.89, 659.86],
    [0.0, 0.0, 1.0]
])
camera_model.set(intrinsic_mat,w,h)
distort_params=np.array([-0.50578, 0.10144, -0.00745, 0.00664, 0.0])
camera_model.set_distort_params(distort_params)
test3d_point=np.array([10.0, 2.0, 20.0])
out_point=camera_model.project(test3d_point)
test2d_point=np.array([123.0,456.0])
d3_out=camera_model.unproject(test2d_point)
print("original:")
print(test3d_point)
print("after:")
print(out_point)
print("before:")
print(test2d_point)
print("final:")
print(d3_out)

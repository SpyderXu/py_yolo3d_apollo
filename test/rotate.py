from __future__ import division
import numpy as np
import math
def eulerAnglesToRotationMatrix(theta):
    R_y = np.array([[math.cos(theta[0]), 0, math.sin(theta[0])],
                    [0, 1, 0],
                    [-math.sin(theta[0]), 0, math.cos(theta[0])]])
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[1]), -math.sin(theta[1])],
                    [0, math.sin(theta[1]), math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
alpha_deg=4.09327559364
theta=np.array([alpha_deg/180.0*math.pi,0,0])
R=eulerAnglesToRotationMatrix(theta)
transform=np.eye(4)
transform[:3,:3]=R
corner=np.array([2.0675571,0.75,0.83009553,1.0])
corner=np.dot(transform,corner)
print(transform)
print("corner:")
print(corner)
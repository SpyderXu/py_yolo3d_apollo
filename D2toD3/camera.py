from __future__ import division
import numpy as np
class CameraModel:
    def __init__(self):
        self.focal_length_x=1
        self.focal_length_y=1
        self.center_x=0
        self.center_y=0
        self.intrinsic=np.zeros((3,3))
        self.intrinsic[0][0]=1
        self.intrinsic[1][1]=1
        self.intrinsic[2][2]=1
        self.width=1
        self.height=1
    def set(self,intrinsic_mat,w,h):
        self.intrinsic=intrinsic_mat
        self.width=w
        self.height=h
        self.focal_length_x=intrinsic_mat[0][0]
        self.focal_length_y=intrinsic_mat[1][1]
        self.center_x=intrinsic_mat[0][2]
        self.center_y=intrinsic_mat[1][2]
    def get_width(self):
        return self.width
    def get_height(self):
        return self.height
    def pixel_denormalize(self,pt2d):
        p=np.zeros(2)
        p[0]=pt2d[0]*self.focal_length_x+self.center_x
        p[1]=pt2d[1]*self.focal_length_y+self.center_y
        return p
    def pixel_normalize(self,pt2d):
        p=np.zeros(2)
        p[0]=(pt2d[0]-self.center_x)/self.focal_length_x
        p[1]=(pt2d[1]-self.center_y)/self.focal_length_y
        return p
    def project(self,pt3d):
        pt2d=np.zeros(2)
        pt2d[0]=pt3d[0]/pt3d[2]
        pt2d[1]=pt3d[1]/pt3d[2]
        return self.pixel_denormalize(pt2d)
    def unproject(self,pt2d):
        pt3d=np.zeros(3)
        pt2d_tmp=self.pixel_normalize(pt2d)
        pt3d[0]=pt2d_tmp[0]
        pt3d[1]=pt2d_tmp[1]
        pt3d[2]=1
        return pt3d




class CameraDistort(CameraModel):
    def __init__(self):
        CameraModel.__init__(self)
        self.distort_params=np.zeros(5)
    def set_distort_params(self,d0,d1,d2,d3,d4):
        self.distort_params[0]=d0
        self.distort_params[1]=d1
        self.distort_params[2]=d2
        self.distort_params[3]=d3
        self.distort_params[4]=d4
    def set_distort_params(self,params):
        self.distort_params=params
    def pixel_normalize(self,pt2d):
        pt2d_distort=CameraModel.pixel_normalize(self,pt2d)
        pt2d_undistort=pt2d_distort.copy()
        for i in range(0,20):
            r_sq=pt2d_undistort[0]*pt2d_undistort[0]+pt2d_undistort[1]*pt2d_undistort[1]
            k_radial=1.0+self.distort_params[0]*r_sq+self.distort_params[1]*r_sq*r_sq+self.distort_params[4]*r_sq*r_sq*r_sq
            delta_x_0=2*self.distort_params[2]*pt2d_undistort[0]*pt2d_undistort[1]+self.distort_params[3]*(r_sq+2*pt2d_undistort[0]*pt2d_undistort[0])
            delta_x_1=self.distort_params[2]*(r_sq+2*pt2d_undistort[1]*pt2d_undistort[1])+2*self.distort_params[3]*pt2d_undistort[0]*pt2d_undistort[1]
            pt2d_undistort[0]=(pt2d_distort[0]-delta_x_0)/k_radial
            pt2d_undistort[1]=(pt2d_distort[1]-delta_x_1)/k_radial
        return pt2d_undistort
    def pixel_denormalize(self,pt2d):
        r_sq=pt2d[0]*pt2d[0]+pt2d[1]*pt2d[1]
        pt2d_radial=pt2d*(1+self.distort_params[0]*r_sq+self.distort_params[1]*r_sq*r_sq+self.distort_params[4]*r_sq*r_sq*r_sq)
        dpt2d=np.zeros(2)
        dpt2d[0]=2 * self.distort_params[2] * pt2d[0] * pt2d[1] + self.distort_params[3] * (r_sq + 2 * pt2d[0] * pt2d[0])
        dpt2d[1] = self.distort_params[2] * (r_sq + 2 * pt2d[1] * pt2d[1]) + 2 * self.distort_params[3] * pt2d[0] * pt2d[1]
        pt2d_undistort=np.zeros(2)
        pt2d_undistort[0] = pt2d_radial[0] + dpt2d[0]
        pt2d_undistort[1] = pt2d_radial[1] + dpt2d[1]
        return CameraModel.pixel_denormalize(self,pt2d_undistort)




































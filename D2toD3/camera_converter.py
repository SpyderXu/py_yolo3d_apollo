from __future__ import division
import numpy as np
import math
from camera import CameraDistort

class GeometryCameraConverter(object):
    def __init__(self):
        self.kMaxDistanceSearchDepth=10
        self.KMaxCenterDirectionSearchDepth=5
        self.camera_model=CameraDistort()
        self.pixel_corners_=np.zeros((8,2))
        self.corners_=np.zeros((8,3))

    def init_camera_model(self,intrinsic_mat,w,h,distort_params):
        """
        :param intrinsic_mat:
        :param w:
        :param h:
        :param distort_params:
        :return:
        """
        self.camera_model.set(intrinsic_mat,w,h)
        self.camera_model.set_distort_params(distort_params)

    def eulerAnglesToRotationMatrix(self,theta):
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

    def rotate(self,alpha_deg,corners):
        theta=np.array([alpha_deg/180.0*math.pi,0,0])
        R = self.eulerAnglesToRotationMatrix(theta)
        transform = np.eye(4)
        transform[:3, :3] = R
        for i in range(0,corners.shape[0]):
            corner=corners[i]
            corner_tmp=np.array([corner[0],corner[1],corner[2],1.0])
            corner_tmp = np.dot(transform, corner_tmp)
            corners[i]=np.array([corner_tmp[0],corner_tmp[1],corner_tmp[2]])
        return corners

    def CheckTruncation(self,obj,trunc_center_pixel):
        width=self.camera_model.get_width()
        height=self.camera_model.get_height()

        if obj.xmin<30.0 or width-30.0<obj.xmax:
            obj.trunc_width=0.5

            if obj.xmin<30.0:
                trunc_center_pixel[0]=obj.xmin
            else:
                trunc_center_pixel[0]=obj.xmax

        if obj.ymin<30.0 or height-30.0<obj.ymax:
            obj.trunc_height=0.5
            trunc_center_pixel[0]=(obj.xmin+obj.ymin)/2.0
        trunc_center_pixel[1]=(obj.ymax+obj.ymin)/2.0
        return trunc_center_pixel

    def CheckSizeSanity(self,obj):
        if int(obj.type) == 0:
            obj.length = max(obj.length, 3.6)
            obj.width = max(obj.width, 1.6 )
            obj.height = max(obj.height, 1.5)
        elif int(obj.type) == 1:
            obj.length = max(obj.length, 0.5)
            obj.width = max(obj.width, 0.5)
            obj.height = max(obj.height, 1.7)
        elif int(obj.type) == 2:
            obj.length = max(obj.length, 1.8)
            obj.width = max(obj.width, 1.2)
            obj.height = max(obj.height, 1.5)
        else:
            obj.length = max(obj.length, 0.5)
            obj.width = max(obj.width, 0.5)
            obj.height = max(obj.height, 1.5)

    def MakeUnit(self,v):
        unit_v=v.copy()
        to_unit_scale=math.sqrt(unit_v[0]*unit_v[0]+unit_v[1]*unit_v[1]+unit_v[2]*unit_v[2])
        unit_v=unit_v/to_unit_scale
        return unit_v

    def SearchDistance(self,pixel_length,use_width,mass_center_v,close_d,far_d):
        curr_d=0.0
        depth=0
        while close_d<=far_d and depth<self.kMaxDistanceSearchDepth:
            # curr_d: current depth
            curr_d=(far_d+close_d)/2.0
            # curr_p: current position
            curr_p=mass_center_v*curr_d
            #print("curr_p:{}".format(curr_p))

            min_p=float("inf")
            max_p=0.0

            for i in range(0,self.corners_.shape[0]):
                point_2d=self.camera_model.project(self.corners_[i]+curr_p)
                #print("point_2d:{}".format(point_2d))
                curr_pixel=0.0

                if use_width:
                    curr_pixel=point_2d[0]
                else:
                    curr_pixel=point_2d[1]
                min_p=min(min_p,curr_pixel)
                max_p=max(max_p,curr_pixel)
            curr_pixel_length=int(max_p-min_p)
            if curr_pixel_length==pixel_length:
                print("obtain curr_pixel_length equal to pixel_length")
                break
            elif pixel_length<curr_pixel_length:
                close_d=curr_d+0.1
            else:
                far_d=curr_d-0.1
            next_d=(far_d+close_d)/2.0
            if abs(next_d-curr_d)<0.1:
                print("early break for 0.1m accuracy")
                break
            depth+=1

        curr_p=mass_center_v*curr_d
        #print("mass_center_v:",mass_center_v.shape)
        for i in range(0,self.corners_.shape[0]):
            point_2d=self.camera_model.project(self.corners_[i]+curr_p)
            self.pixel_corners_[i]=point_2d
        return curr_d

    def SearchCenterDirection(self,box_center_pixel,curr_d,mass_center_v,mass_center_pixel):
        depth=0

        while depth<self.KMaxCenterDirectionSearchDepth:
            new_center_v=mass_center_v*curr_d
            max_pixel_x = float("-inf")
            min_pixel_x = float("inf")
            max_pixel_y = float("-inf")
            min_pixel_y = float("inf")

            for i in range(0,self.corners_.shape[0]):
                point_2d=self.camera_model.project(self.corners_[i]+new_center_v)
                min_pixel_x = min(min_pixel_x, point_2d[0])
                max_pixel_x = max(max_pixel_x, point_2d[0])
                min_pixel_y = min(min_pixel_y, point_2d[1])
                max_pixel_y = max(max_pixel_y, point_2d[1])

            current_box_center_pixel=np.zeros(2)
            current_box_center_pixel[0]=(max_pixel_x+min_pixel_x)/2.0
            current_box_center_pixel[1]=(max_pixel_y+min_pixel_y)/2.0

            print("current box center in search direction:",current_box_center_pixel)
            print("box center pixel in search direction:",box_center_pixel)

            mass_center_pixel[0]=mass_center_pixel[0]+(box_center_pixel[0]-current_box_center_pixel[0])
            mass_center_pixel[1] = mass_center_pixel[1] + (box_center_pixel[1] - current_box_center_pixel[1])
            print("mass center pixel in search direction:",mass_center_pixel)
            mass_center_v=self.camera_model.unproject(mass_center_pixel)
            print("mass center v in search direction:",mass_center_v)
            mass_center_v=self.MakeUnit(mass_center_v)

            if abs(mass_center_pixel[0]-box_center_pixel[0])<1.0 and abs(mass_center_pixel[1]-box_center_pixel[1])<1.0:
                break

            depth+=1
        return

    def DecideAngle(self,camera_ray,obj):
        beta=math.atan2(camera_ray[0],camera_ray[2])
        if obj.distance>60.0 or obj.trunc_width>0.25:
            print("handle in special case")
            obj.theta=-1.0*math.pi/2.0
            obj.alpha=obj.theta-beta
            if obj.alpha>math.pi:
                obj.alpha-=2*math.pi
            elif obj.alpha<-1*math.pi:
                obj.alpha+=2*math.pi
        else:
            theta=obj.alpha+beta
            if theta>math.pi:
                theta-=2*math.pi
            elif theta<-1*math.pi:
                theta+=2*math.pi
            obj.theta=theta
        obj.direction=np.array([math.cos(obj.theta),0.0,-1*math.sin(obj.theta)])

    def SetBoxProjection(self,obj):
        if obj.trunc_width<0.25 and obj.trunc_height<0.25:
            for i in range(8):
                obj.pt8s[i]=self.pixel_corners_[i]
        return

    def convert(self,objects):
        if len(objects)==0:
            return False

        for obj in objects:
            trunc_center_pixel=np.zeros(2)
            trunc_center_pixel=self.CheckTruncation(obj,trunc_center_pixel)
            self.CheckSizeSanity(obj)

            deg_alpha=obj.alpha*180.0/math.pi

            upper_left=np.array([obj.xmin,obj.ymin])
            lower_right=np.array([obj.xmax,obj.ymax])

            distance=0.0
            mass_center_pixel=np.zeros(2)

            #print("obj trunc_height:{}".format(obj.trunc_height))

            if obj.trunc_height<0.25:
                distance=self.convertSingle(obj.height,obj.width,obj.length,deg_alpha,upper_left,lower_right,False,distance,mass_center_pixel)
                #print("no truncation on 2d height")
            elif obj.trunc_width<0.25 and obj.trunc_height>0.25:
                distance=self.convertSingle(obj.height,obj.width,obj.length,deg_alpha,upper_left,lower_right,True,distance,mass_center_pixel)
            else:
                distance=10.0
                mass_center_pixel=trunc_center_pixel
            obj.distance=distance

            camera_ray=self.camera_model.unproject(mass_center_pixel)

            self.DecideAngle(camera_ray,obj)

            scale=obj.distance/math.sqrt(camera_ray[0]*camera_ray[0]+camera_ray[1]*camera_ray[1]+camera_ray[2]*camera_ray[2])

            obj.center=camera_ray*scale

            self.SetBoxProjection(obj)

        return True

    def convertSingle(self,h,w,l,alpha_deg,upper_left,lower_right,use_width,distance,mass_center_pixel):
        # target goals: projection target
        pixel_width=int(lower_right[0]-upper_left[0])
        pixel_height=int(lower_right[1]-upper_left[1])
        pixel_length=pixel_height
        if use_width:
            pixel_length=pixel_width

        # target goals: box center pixel
        box_center_pixel=np.array([(lower_right[0]+upper_left[0])/2.0,(lower_right[1]+upper_left[1])/2.0])

        h_half=h/2.0
        w_half=w/2.0
        l_half=l/2.0

        deg_alpha=alpha_deg
        corners=np.zeros((8,3))
        corners[0] = np.array([l_half, h_half, w_half])
        corners[1] = np.array([l_half, h_half, -1*w_half])
        corners[2] = np.array([-1*l_half, h_half, -1*w_half])
        corners[3] = np.array([-1*l_half, h_half, w_half])
        corners[4] = np.array([l_half, -1*h_half, w_half])
        corners[5] = np.array([l_half, -1*h_half, -1*w_half])
        corners[6] = np.array([-1*l_half, -1*h_half, -1*w_half])
        corners[7] = np.array([-1*l_half, -1*h_half, w_half])

        corners=self.rotate(deg_alpha,corners)
        self.corners_=corners

        middle_v=np.array([0.0,0.0,20.0])
        center_pixel=self.camera_model.project(middle_v)

        max_pixel_x=float("-inf")
        min_pixel_x=float("inf")
        max_pixel_y=float("-inf")
        min_pixel_y=float("inf")

        for i in range(0,corners.shape[0]):
            point_2d=self.camera_model.project(corners[i]+middle_v)
            min_pixel_x=min(min_pixel_x,point_2d[0])
            max_pixel_x=max(max_pixel_x,point_2d[0])
            min_pixel_y=min(min_pixel_y,point_2d[1])
            max_pixel_y=max(max_pixel_y,point_2d[1])

        relative_x=(center_pixel[0]-min_pixel_x)/(max_pixel_x-min_pixel_x)
        relative_y=(center_pixel[1]-min_pixel_y)/(max_pixel_y-min_pixel_y)

        mass_center_pixel[0]=(lower_right[0]-upper_left[0])*relative_x+upper_left[0]
        mass_center_pixel[1]=(lower_right[1]-upper_left[1])*relative_y+upper_left[1]

        mass_center_v=self.camera_model.unproject(mass_center_pixel)
        print("mass_center_pixel before search distance:",mass_center_pixel)
        print("mass_center_v before search distance:",mass_center_v)

        mass_center_v=self.MakeUnit(mass_center_v)

        distance=self.SearchDistance(pixel_length,use_width,mass_center_v,0.1,150.0)

        print("search distance 1 is {}".format(distance))

        for i in range(0,1):
            self.SearchCenterDirection(box_center_pixel,distance,mass_center_v,mass_center_pixel)
            distance=self.SearchDistance(pixel_length,use_width,mass_center_v,0.9*distance,1.1*distance)
            print("search distance 2 is {}".format(distance))
        return distance





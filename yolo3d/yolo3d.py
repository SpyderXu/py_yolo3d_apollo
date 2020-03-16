#from __future__ import division
import numpy as np
import caffe
import cv2
import math
from utils import sigmoid,py_cpu_nms
import time

class obj_3d(object):
    def __init__(self,box):
        self.xmin=box[0]
        self.ymin=box[1]
        self.xmax=box[2]
        self.ymax=box[3]
        self.prob=box[4]
        self.type=box[5]
        self.orientation=box[6]
        self.height=box[7]
        self.width=box[8]
        self.length=box[9]
        self.lof_xmin=box[10]
        self.lof_ymin=box[11]
        self.lof_xmax=box[12]
        self.lof_ymax=box[13]
        self.lor_xmin=box[14]
        self.lor_ymin=box[15]
        self.lor_xmax=box[16]
        self.lor_ymax=box[17]
        self.trunc_width=0.0
        self.trunc_height=0.0
        self.distance=0.0
        self.theta=0.0
        self.alpha=box[6]
        self.center=np.array([0.0,0.0,0.0])
        self.pt8s=np.zeros((8,2))

class Yolo3D(object):
    def __init__(self,config):
        self.deploy_path=config.model.deploy_path
        self.weights_path=config.model.weight_path
        self.anchors=config.anchors
        self.classes=config.classes
        self.num_classes=len(self.classes)
        self.seg_thresh=config.seg_thresh
        self.det_thresh =config.det_thresh
        self.nms_thresh=config.nms_thresh
        self.resize_or_crop=config.resize_or_crop
        self.input_width=config.input_width
        self.input_height=config.input_height
        self.offset_ratio=0.28843
        self.image_width=1920
        self.image_height=1080
        ## for time test
        self.t1=0
        self.t2=0
        self.t3=0

    def init_net(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net(self.deploy_path, self.weights_path, caffe.TEST)

    def init_params(self):
        self.offset_y=int(self.offset_ratio*self.image_height+0.5)
        self.roi_h=self.image_height-self.offset_y
        self.crop_img_width=self.image_width
        self.crop_img_height=self.roi_h

    def change2objs(self,boxes):
        ret_obj=[]
        for box in boxes:
            ret_obj.append(obj_3d(box))
        return ret_obj

    def obtain_boxes(self,index,row,col,n,width,height,prob,label):
        ori_index = index * 2
        orientation = math.atan2(self.ori_blob[index + 1], self.ori_blob[index])

        dim_index = index * 3
        d3_h, d3_w, d3_l = self.dim_blob[dim_index:dim_index + 3]

        box_index = index * 4
        cx = (col + sigmoid(self.loc_blob[box_index + 0])) / (width * 1.0) * self.crop_img_width
        cy = (row + sigmoid(self.loc_blob[box_index + 1])) / (height * 1.0) * self.crop_img_height
        w = math.exp(self.loc_blob[box_index + 2]) * self.anchors[n, 0] / (width * 1.0) * 0.5 * self.crop_img_width
        h = math.exp(self.loc_blob[box_index + 3]) * self.anchors[n, 1] / (height * 1.0) * 0.5 * self.crop_img_height

        lof_index = index * 4
        lof_x = self.lof_blob[lof_index + 0] * w * 2 + cx
        lof_y = self.lof_blob[lof_index + 1] * h * 2 + cy
        lof_w = math.exp(self.lof_blob[lof_index + 2]) * w
        lof_h = math.exp(self.lof_blob[lof_index + 3]) * h

        lor_index = index * 4
        lor_x = self.lor_blob[lor_index + 0] * w * 2 + cx
        lor_y = self.lor_blob[lor_index + 1] * h * 2 + cy
        lor_w = math.exp(self.lor_blob[lor_index + 2]) * w
        lor_h = math.exp(self.lor_blob[lor_index + 3]) * h

        return np.array([cx-w,cy-h+self.offset_y,
                         cx+w,cy+h+self.offset_y,
                         prob,
                         label,
                         orientation,
                         d3_h, d3_w, d3_l,
                         lof_x-lof_w,lof_y-lof_h+self.offset_y,
                         lof_x+lof_w,lof_y+lof_h+self.offset_y,
                         lor_x-lor_w,lor_y-lor_h+self.offset_y,
                         lor_x+lor_w,lor_y+lor_h+self.offset_y])

    def transform_boxes_v2(self, num_classes, im):

        batch = self.obj_blob.shape[0]
        height = self.obj_blob.shape[1]
        width = self.obj_blob.shape[2]
        num_anchors = self.anchors.shape[0]

        self.obj_blob = self.obj_blob.reshape(-1)
        self.cls_blob = self.cls_blob.reshape(-1)
        self.loc_blob = self.loc_blob.reshape(-1)
        self.ori_blob = self.ori_blob.reshape(-1)
        self.dim_blob = self.dim_blob.reshape(-1)
        self.lof_blob = self.lof_blob.reshape(-1)
        self.lor_blob = self.lor_blob.reshape(-1)

        ret_list = []
        self.t1 = time.time()
        for i in xrange(height * width):
            row = i / width
            col = i % width
            for n in range(num_anchors):
                index = i * num_anchors + n
                scale = self.obj_blob[index]

                class_index = index * num_classes
                for k in range(0, num_classes):
                    prob = scale * self.cls_blob[class_index + k]
                    if prob > self.det_thresh:
                        ret_list.append(self.obtain_boxes(index,row,col,n,width,height,prob,k))
        self.t2 = time.time()
        print("t:1-2:{}".format(self.t2 - self.t1))
        if len(ret_list) != 0:
            ret_list = np.array(ret_list, dtype=np.float32)
            keep = py_cpu_nms(ret_list, self.nms_thresh)
            ret_list = ret_list[keep]
        self.t3=time.time()
        print("t2-3:{}".format(self.t3-self.t2))
        return self.change2objs(ret_list)

    def transform_boxes(self, num_classes, im):

        batch = self.obj_blob.shape[0]
        height = self.obj_blob.shape[1]
        width = self.obj_blob.shape[2]
        num_anchors = self.anchors.shape[0]

        obj_pred = self.obj_blob.reshape(-1)
        cls_pred = self.cls_blob.reshape(-1)
        loc_pred = self.loc_blob.reshape(-1)
        ori_pred = self.ori_blob.reshape(-1)
        dim_pred = self.dim_blob.reshape(-1)
        lof_pred = self.lof_blob.reshape(-1)
        lor_pred = self.lor_blob.reshape(-1)

        ret_list = []
        self.t1 = time.time()
        for i in xrange(height * width):
            row = i / width
            col = i % width
            for n in range(num_anchors):
                obj_np = np.zeros(18)
                index = i * num_anchors + n
                scale = obj_pred[index]

                ori_index = index * 2
                orientation = math.atan2(ori_pred[index + 1], ori_pred[index])

                dim_index = index * 3
                d3_h = dim_pred[dim_index + 0]
                d3_w = dim_pred[dim_index + 1]
                d3_l = dim_pred[dim_index + 2]

                box_index = index * 4
                cx = (col + sigmoid(loc_pred[box_index + 0])) / (width * 1.0)
                cy = (row + sigmoid(loc_pred[box_index + 1])) / (height * 1.0)
                w = math.exp(loc_pred[box_index + 2]) * self.anchors[n, 0] / (width * 1.0) * 0.5
                h = math.exp(loc_pred[box_index + 3]) * self.anchors[n, 1] / (height * 1.0) * 0.5

                lof_index = index * 4
                lof_x = lof_pred[lof_index + 0] * w * 2 + cx
                lof_y = lof_pred[lof_index + 1] * h * 2 + cy
                lof_w = math.exp(lof_pred[lof_index + 2]) * w
                lof_h = math.exp(lof_pred[lof_index + 3]) * h

                lor_index = index * 4
                lor_x = lor_pred[lor_index + 0] * w * 2 + cx
                lor_y = lor_pred[lor_index + 1] * h * 2 + cy
                lor_w = math.exp(lor_pred[lor_index + 2]) * w
                lor_h = math.exp(lor_pred[lor_index + 3]) * h

                cx = self.crop_img_width * cx
                cy = self.crop_img_height * cy
                w = self.crop_img_width * w
                h = self.crop_img_height * h

                lof_x = self.crop_img_width * lof_x
                lof_y = self.crop_img_height * lof_y
                lof_w = self.crop_img_width * lof_w
                lof_h = self.crop_img_height * lof_h

                lor_x = self.crop_img_width * lor_x
                lor_y = self.crop_img_height * lor_y
                lor_w = self.crop_img_width * lor_w
                lor_h = self.crop_img_height * lor_h

                class_index = index * num_classes
                for k in range(0, num_classes):
                    prob = scale * cls_pred[class_index + k]
                    if prob > self.det_thresh:
                        obj_np[0] = cx - w
                        obj_np[1] = cy - h + self.offset_y
                        obj_np[2] = cx + w
                        obj_np[3] = cy + h + self.offset_y
                        obj_np[4] = prob
                        obj_np[5] = k
                        obj_np[6] = orientation
                        obj_np[7] = d3_h
                        obj_np[8] = d3_w
                        obj_np[9] = d3_l
                        obj_np[10] = lof_x - lof_w
                        obj_np[11] = lof_y - lof_h + self.offset_y
                        obj_np[12] = lof_x + lof_w
                        obj_np[13] = lof_y + lof_h + self.offset_y
                        obj_np[14] = lor_x - lor_w
                        obj_np[15] = lor_y - lor_h + self.offset_y
                        obj_np[16] = lor_x + lor_w
                        obj_np[17] = lor_y + lor_h + self.offset_y

                        ret_list.append(obj_np)
        self.t2 = time.time()
        print("t:1-2:{}".format(self.t2 - self.t1))
        if len(ret_list) != 0:
            ret_list = np.array(ret_list, dtype=np.float32)
            keep = py_cpu_nms(ret_list, self.nms_thresh)
        ret_list = ret_list[keep]
        self.t3=time.time()
        print("t2-3:{}".format(self.t3-self.t2))
        return self.change2objs(ret_list)

    def process_img(self,im):

        if self.resize_or_crop=="resize":
            self.crop_img_height,self.crop_img_width,_=im.shape
            self.offset_y=0
            im=cv2.resize(im,(self.input_width,self.input_height))
        else:
            im=im[self.offset_y:,:,:]
            im=cv2.resize(im,(self.input_width,self.input_height))
        im = np.array(im, dtype=np.float32)
        self.net.blobs['data'].data[...] = im
        self.net.forward()

        lane = self.net.blobs["seg_prob"].data
        self.obj_blob = self.net.blobs["obj_pred"].data
        self.cls_blob = self.net.blobs["cls_pred"].data
        self.loc_blob = self.net.blobs["loc_pred"].data
        self.ori_blob = self.net.blobs["ori_pred"].data
        self.dim_blob = self.net.blobs["dim_pred"].data
        self.lof_blob = self.net.blobs["lof_pred"].data
        self.lor_blob = self.net.blobs["lor_pred"].data


        return self.transform_boxes_v2(self.num_classes, im)








import os
import caffe
import cv2
import numpy as np
import math

num_anchors = 16
num_classes = 4
anchors = np.array([
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

thresh = 0.5
obj_thresh = 0.8
nms_thresh = 0.45
img_name = "ceshi.jpg"
rootDir = "/home/cvpr/dataset/yolo_camera_detector/yolo3d_1128"
deploy_file = os.path.join(rootDir, "deploy.pt")
weights_file = os.path.join(rootDir, "deploy.md")
net = caffe.Net(deploy_file, weights_file, caffe.TEST)

im = cv2.imread(img_name)
im = cv2.resize(im, (960, 384))
orig = im.copy()
# cv2.imshow("img",im)
im = np.array(im, dtype=np.float32)
net.blobs['data'].data[...] = im
net.forward()
lane = net.blobs["seg_prob"].data
obj_blob = net.blobs["obj_pred"].data
cls_blob = net.blobs["cls_pred"].data
loc_blob = net.blobs["loc_pred"].data
ori_blob = net.blobs["ori_pred"].data
dim_blob = net.blobs["dim_pred"].data
lof_blob = net.blobs["lof_pred"].data
lor_blob = net.blobs["lor_pred"].data


def sigmoid(x):
    return 1.0 / (math.exp(-1.0 * x) + 1)

def draw_3d_box(im,lof,lor):
    lof_xmin,lof_ymin,lof_xmax,lof_ymax=lof
    lor_xmin,lor_ymin,lor_xmax,lor_ymax=lor
    cv2.rectangle(im,(lof_xmin,lof_ymin),(lof_xmax,lof_ymax),(255,0,255),1)
    cv2.rectangle(im,(lor_xmin,lor_ymin),(lor_xmax,lor_ymax),(255,0,255),1)
    cv2.line(im,(lof_xmin,lof_ymin),(lor_xmin,lor_ymin),(0,255,0),1)
    cv2.line(im,(lof_xmax,lof_ymax),(lor_xmax,lor_ymax),(0,255,0),1)
    cv2.line(im,(lof_xmin,lof_ymax),(lor_xmin,lor_ymax),(0,255,0),1)
    cv2.line(im,(lof_xmax,lof_ymin),(lor_xmax,lor_ymin),(0,255,0),1)

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def transform_boxes(obj_blob, cls_blob, loc_blob, num_classes, anchors, im):
    batch = obj_blob.shape[0]
    height = obj_blob.shape[1]
    width = obj_blob.shape[2]
    num_anchors = anchors.shape[0]

    img_h, img_w, _ = im.shape
    obj_pred = obj_blob.reshape(-1)
    cls_pred = cls_blob.reshape(-1)
    loc_pred = loc_blob.reshape(-1)
    ori_pred = ori_blob.reshape(-1)
    dim_pred = dim_blob.reshape(-1)
    lof_pred = lof_blob.reshape(-1)
    lor_pred = lor_blob.reshape(-1)

    ret_list = []
    for i in xrange(height * width):
        row = i / width
        col = i % width
        for n in range(num_anchors):
            obj_np = np.zeros(18)
            index = i * num_anchors + n
            scale = obj_pred[index]

            ori_index=index*2
            orientation = math.atan2(ori_pred[index+1], ori_pred[index])

            dim_index=index*3
            d3_h=dim_pred[dim_index+0]
            d3_w=dim_pred[dim_index+1]
            d3_l=dim_pred[dim_index+2]

            box_index = index * 4
            cx = (col + sigmoid(loc_pred[box_index + 0])) / (width * 1.0)
            cy = (row + sigmoid(loc_pred[box_index + 1])) / (height * 1.0)
            w = math.exp(loc_pred[box_index + 2]) * anchors[n, 0] / (width * 1.0)*0.5
            h = math.exp(loc_pred[box_index + 3]) * anchors[n, 1] / (height * 1.0)*0.5

            print("cx:{},cy:{},w:{},h:{}".format(cx, cy, w, h))


            lof_index = index * 4
            lof_x = lof_pred[lof_index+0]*w*2+cx
            lof_y = lof_pred[lof_index+1]*h*2+cy
            lof_w = math.exp(lof_pred[lof_index+2])*w
            lof_h = math.exp(lof_pred[lof_index+3])*h

            lor_index = index * 4
            lor_x = lor_pred[lor_index + 0] * w * 2 + cx
            lor_y = lor_pred[lor_index + 1] * h * 2 + cy
            lor_w = math.exp(lor_pred[lor_index + 2]) * w
            lor_h = math.exp(lor_pred[lor_index + 3]) * h

            cx = img_w * cx
            cy = img_h * cy
            w = img_w * w
            h = img_h * h

            lof_x = img_w * lof_x
            lof_y = img_h * lof_y
            lof_w = img_w * lof_w
            lof_h = img_h * lof_h

            lor_x = img_w * lor_x
            lor_y = img_h * lor_y
            lor_w = img_w * lor_w
            lor_h = img_h * lor_h


            class_index = index * num_classes
            for k in range(0, num_classes):
                prob = scale * cls_pred[class_index + k]
                if prob > obj_thresh:
                    obj_np[0] = cx - w
                    obj_np[1] = cy - h
                    obj_np[2] = cx + w
                    obj_np[3] = cy + h
                    obj_np[4] = prob
                    obj_np[5] = k
                    obj_np[6] = orientation
                    obj_np[7] = d3_w
                    obj_np[8] = d3_h
                    obj_np[9] = d3_l
                    obj_np[10] = lof_x-lof_w
                    obj_np[11] = lof_y-lof_h
                    obj_np[12] = lof_x+lof_w
                    obj_np[13] = lof_y+lof_h
                    obj_np[14] = lor_x - lor_w
                    obj_np[15] = lor_y - lor_h
                    obj_np[16] = lor_x + lor_w
                    obj_np[17] = lor_y + lor_h

                    ret_list.append(obj_np)
    if len(ret_list) != 0:
        ret_list = np.array(ret_list, dtype=np.float32)
        keep=py_cpu_nms(ret_list, nms_thresh)
    ret_list=ret_list[keep]

    for i in range(0, len(ret_list)):
        obj=ret_list[i]
        obj_xmin=int(obj[0])
        obj_ymin=int(obj[1])
        obj_xmax=int(obj[2])
        obj_ymax=int(obj[3])
        lof_xmin=int(obj[10])
        lof_ymin=int(obj[11])
        lof_xmax=int(obj[12])
        lof_ymax=int(obj[13])
        lor_xmin = int(obj[14])
        lor_ymin = int(obj[15])
        lor_xmax = int(obj[16])
        lor_ymax = int(obj[17])
        obj_cx=int((obj_xmin+obj_xmax)/2.0)
        obj_cy=int((obj_ymin+obj_ymax)/2.0)
        cv2.rectangle(im,(obj_xmin,obj_ymin),(obj_xmax,obj_ymax),(0,0,255),1)
        #draw_3d_box(im,(lof_xmin,lof_ymin,lof_xmax,lof_ymax),(lor_xmin,lor_ymin,lor_xmax,lor_ymax))

        #cv2.putText(im, str(int(obj[5])), (obj_cx, obj_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #cv2.putText(im, str(round(obj[6],2)), (obj_cx, obj_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj[7],1)), (obj_cx, obj_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj[8],1)), (obj_cx, obj_cy+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj[9],1)), (obj_cx, obj_cy+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return im


orig = transform_boxes(obj_blob, cls_blob, loc_blob, num_classes, anchors, orig)
#lane0=lane[0,0,:,:]
#cv2.imshow("lane",np.array((lane0>0.5)*255,dtype=np.uint8))
cv2.imshow("im", orig)
cv2.waitKey(0)

print("lof")
print(lof_blob.shape)
print("lor")
print(lor_blob.shape)
print("lane")
print(lane.shape)


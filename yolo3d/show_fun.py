import numpy as np
import cv2

def draw_axis(im):
    cv2.line(im, (0, 980), (1000, 980), (0, 0, 255), 2)
    cv2.line(im, (500, 0), (500, 1000), (0, 0, 255), 2)
def show_fov(im,objs):
    for obj in objs:
        z=int(obj.center[2])
        x=int(obj.center[0])
        cv2.circle(im,(500+10*x,980-z*10),10,(255,0,0))

def draw_2d_box(im,objs):
    for obj in objs:
        cv2.rectangle(im,(int(obj.xmin),int(obj.ymin)),(int(obj.xmax),int(obj.ymax)),(0,0,255),1)
    return

def draw_25d_box(im,objs):
    for obj in objs:
        cv2.rectangle(im,(int(obj.xmin),int(obj.ymin)),(int(obj.xmax),int(obj.ymax)),(0,0,255),1)
        #cv2.putText(im, str(round(obj.distance, 2)), (int((obj.xmin+obj.xmax)/2.0), int((obj.ymin+obj.ymax)/2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj.center[0], 2)),
                    (int((obj.xmin + obj.xmax) / 2.0), int((obj.ymin + obj.ymax) / 2.0)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        cv2.putText(im, str(round(obj.center[1], 2)),
                    (int((obj.xmin + obj.xmax) / 2.0), int((obj.ymin + obj.ymax) / 2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        cv2.putText(im, str(round(obj.center[2], 2)),
                    (int((obj.xmin + obj.xmax) / 2.0), int((obj.ymin + obj.ymax) / 2.0)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
    return

def draw_3d_box(im,objs):
    for obj in objs:
        cv2.rectangle(im, (int(obj.lof_xmin), int(obj.lof_ymin)), (int(obj.lof_xmax), int(obj.lof_ymax)), (255, 255, 255), 1)
        cv2.rectangle(im, (int(obj.lor_xmin), int(obj.lor_ymin)), (int(obj.lor_xmax), int(obj.lor_ymax)), (255, 0, 255), 1)
        cv2.line(im, (int(obj.lof_xmin), int(obj.lof_ymin)), (int(obj.lor_xmin), int(obj.lor_ymin)), (0, 255, 0), 1)
        cv2.line(im, (int(obj.lof_xmax), int(obj.lof_ymax)), (int(obj.lor_xmax), int(obj.lor_ymax)), (0, 255, 0), 1)
        cv2.line(im, (int(obj.lof_xmin), int(obj.lof_ymax)), (int(obj.lor_xmin), int(obj.lor_ymax)), (0, 255, 0), 1)
        cv2.line(im, (int(obj.lof_xmax), int(obj.lof_ymin)), (int(obj.lor_xmax), int(obj.lor_ymin)), (0, 255, 0), 1)

def draw_3dim(im,objs):
    for obj in objs:
        cv2.putText(im, str(round(obj.height, 2)), (int((obj.xmin+obj.xmax)/2.0), int((obj.ymin+obj.ymax)/2.0)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj.width, 2)), (int((obj.xmin+obj.xmax)/2.0), int((obj.ymin+obj.ymax)/2.0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(im, str(round(obj.length, 2)), (int((obj.xmin+obj.xmax)/2.0), int((obj.ymin+obj.ymax)/2.0)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
"""
def draw_3d_box(im,lof,lor):
    lof_xmin,lof_ymin,lof_xmax,lof_ymax=lof
    lor_xmin,lor_ymin,lor_xmax,lor_ymax=lor
    cv2.rectangle(im,(lof_xmin,lof_ymin),(lof_xmax,lof_ymax),(255,0,255),1)
    cv2.rectangle(im,(lor_xmin,lor_ymin),(lor_xmax,lor_ymax),(255,0,255),1)
    cv2.line(im,(lof_xmin,lof_ymin),(lor_xmin,lor_ymin),(0,255,0),1)
    cv2.line(im,(lof_xmax,lof_ymax),(lor_xmax,lor_ymax),(0,255,0),1)
    cv2.line(im,(lof_xmin,lof_ymax),(lor_xmin,lor_ymax),(0,255,0),1)
    cv2.line(im,(lof_xmax,lof_ymin),(lor_xmax,lor_ymin),(0,255,0),1)
"""

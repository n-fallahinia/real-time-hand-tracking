# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS

import cv2
import imutils
import math

def vidCapture(video, size, verbos=False):
    # raeding the frame
    cap_success, frame = video.read()
    if not cap_success:
        return  
    image_np = imutils.resize(frame, width=size[0])
    (H, W) = image_np.shape[:2]
    if verbos:
        print("frame size is: {0:2d} * {1:2d}".format(W, H))
    return image_np

def vidRecord(frame_size, rate=20.0):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    vid_save = cv2.VideoWriter('output.avi',fourcc, rate, frame_size)
    return vid_save

def boxProcess(boxes, image, verbose=False):
    idx = 1
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if verbose:
            print("box {0:2d}".format(idx))
            print("\tX :{0:2d}, y :{1:2d}, W :{2:2d}, H :{0:2d},".format(x, y, w, h))
        idx += 1
    return image

def video_crop_from_frame(image_np, box):
    x, y, w, h = box
    ymin = y
    ymax = y + h
    xmin = x 
    xmax = x + w
    (left, right, top, bottom) = ( math.floor(xmin), math.floor(xmax), math.floor(ymin), math.floor(ymax) )  
    return image_np[top:bottom, left:right]

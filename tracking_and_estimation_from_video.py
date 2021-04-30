""" Script to run a real-time multi-finger tracking along with 
    the image alignment and 3D force estimation model from a video 

    Navid Fallahinia - 02/27/2021
    BioRobotics Lab

usage: tracking_and_estimation_from_video.py [-v VIDEO_DIR] [-t TRACKER] [-e ESTIMATION_MDL] [-d DETECTION_MDL]
"""
import os
import io
import argparse
import imutils
import time

import cv2
import numpy as np
import tensorflow as tf
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from utils.estimationutils import *
from utils.videoutils import *

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="real-time multi-finger tracking and force estimation")

parser.add_argument("-v", "--input_video", type=str, default='vid.avi',
                    help="Path to input video file")

parser.add_argument("-t", "--tracker", type=str, default="kcf",
                    help="Object tracker model")

parser.add_argument("-e", "--estimation_model", type=str, default='./inference_model_estimation/model_1',
                    help="Path to the trained force estimation and alignment model")

parser.add_argument("-d", "--detection_model", type=str, default="./inference_model_detection/inference_graph_1",
                    help="Path to the trained finger detection model")

parser.add_argument("-n", "--finger_number", type=int, default=3,
                    help="the maximum number of fingers in each frame to be found")

parser.add_argument("-l", "--label_dir", help="Path to the label map file.", default='./inference_model_detection/labelmap.pbtxt',
                    type=str)
args = parser.parse_args()

# object tracker models. KCF is the default model. MIL is also recommended however TLD and CSRT are not recommended
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create, }

FINGERS = {
    0 : "Index finger",
    1: "Middle finger",
    2: "Ring finger",
    3: "Thumb", }

if __name__ == '__main__':

    # Enable GPU dynamic memory allocation
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    PATH_TO_TRACKING_SAVED_MODEL = args.detection_model + "/saved_model"
    PATH_TO_LABELS = args.label_dir
    PATH_TO_ESTIMATION_SAVED_MODEL = args.estimation_model

    # check if the detection model is available
    assert os.path.exists(PATH_TO_TRACKING_SAVED_MODEL), "No detection model found at {}".format(
        args.detection_model)
    # check if the detection model is available
    assert os.path.exists(PATH_TO_ESTIMATION_SAVED_MODEL), "No estimation model found at {}".format(
        args.estimation_model)

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)

    # loading the estimation and the detection models from the saved directory
    print('Loading the detection model...', end='')
    start_time = time.time()
    # detection_model = tf.saved_model.load(PATH_TO_TRACKING_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    print('Loading the estimation model...', end='')
    start_time = time.time()
    estimation_model = tf.saved_model.load(PATH_TO_ESTIMATION_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    print('=================================================')

    desired_object_num = args.finger_number
    frame_size = (600, 853)

    # grab the appropriate object tracker using our dictionary of
    trackers = cv2.legacy.MultiTracker_create()
    video_obj = cv2.VideoCapture(args.input_video)
    first_frame = vidCapture(video_obj, frame_size, verbos=True)

    print("Starting the video feed...")

    # initial bounding boxes for the number of fingers. This part will be replaced with the detector later
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    init_boxes_np = np.array([[177, 186, 143, 117],
                              [184, 380, 165, 113],
                              [146, 537, 156, 107]], dtype=int)
    # init_boxes_np = cv2.selectROIs("Initial Frame", first_frame,
    #                             fromCenter=False, showCrosshair=True)
    init_boxes = tuple(map(tuple, init_boxes_np))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++

    # updating the tracking with initial bounding boxes
    print("\t{:2d} initial fingers have been detetcted".format(len(init_boxes)))
    for found_object_num in range(len(init_boxes)):
        tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
        trackers.add(tracker, first_frame, init_boxes[found_object_num])
        success, boxes = trackers.update(first_frame)
    # loop over frames from the video stream
    while True:
        fps = FPS().start()
        frame_np = vidCapture(video_obj, frame_size)
        # check if the captured frame does exist
        if frame_np is None:
            print("No frame found at {:.2f}".format(fps.fps()))
            break
        # update the box locations in each new frame
        success, boxes = trackers.update(frame_np)
        if success:
            fps.update()
            fps.stop()
            frame_np = boxProcess(boxes, frame_np)
            info = [("Tracker", args.tracker), ("FPS", "{:.2f}".format(fps.fps()))]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame_np, text, (10, frame_size[1] - (
                            (i * 20) + 20)),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame_np)
        key = cv2.waitKey(20) & 0xFF
        # crop the fingers in each frame
        for box_idx, box in enumerate(boxes):
            img_cropped = video_crop_from_frame(frame_np, box)
            estimated_forces = force_from_image_numpy(img_cropped, estimation_model)
            print("\t{:}: fx= {:.2f} | fy={:.2f} | fz={:.2f}".format(FINGERS[box_idx],
                    estimated_forces[0], estimated_forces[1], estimated_forces[2]))
            cv2.imshow("finger {:2d}".format(box_idx), img_cropped)
        print('-----------------------------------------------------')

        # stop the video
        if key == ord("q"):
            break

    video_obj.release()
    cv2.destroyAllWindows()

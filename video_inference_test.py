""" Simple script to run a simple inference test from a video 

Navid Fallahinia - 21/12/2020
BioRobotics Lab

usage: inference_test.py [-h] [-tdir MDL_DIR] [-l LABELS_PATH] [-fdir EST_DIR]

"""

import os
import io
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from utils.inferenceutils import *

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample inference test from live camera feed ")
parser.add_argument("-tdir",
                    "--model_dir",
                    help="Path to the folder where the model is stored.",
                    default='./inference_model_detection/inference_graph_1',
                    type=str)
parser.add_argument("-l",
                    "--label_dir",
                    help="Path to the label map file.",
                    default='./inference_model_detection/labelmap.pbtxt',
                    type=str)
parser.add_argument("-fdir",
                    "--estimation_dir",
                    help="Path to the folder where force estimation model is stored.",
                    # default='./inference_model/labelmap.pbtxt',
                    type=str)
args = parser.parse_args()
 
if __name__ == '__main__':

    # Enable GPU dynamic memory allocation
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)
    # some variables
    PATH_TO_TRACKING_SAVED_MODEL = args.model_dir + "/saved_model"
    PATH_TO_LABELS = args.label_dir
    PATH_TO_VIDEO = '/video/vid.avi'
    min_score_thresh = 0.4
    # loading the detection model 
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print('Loading the detection model...', end='')
    start_time = time.time()
    detection_model = tf.saved_model.load(PATH_TO_TRACKING_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    # create the tracker object nad the video reader
    tracker = cv2.TrackerKCF_create()
    cap = cv2.VideoCapture('vid.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_dict = run_inference_for_single_image(detection_model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            min_score_thresh=min_score_thresh,
            line_thickness=3)

        num_box_images = len([i for i in output_dict['detection_scores'] if i > min_score_thresh])
        # if num_box_images == 3:
            
            # find_abstract_boxes()

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        cv2.imshow('frame',image_np)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
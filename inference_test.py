""" Simple script to run a simple inference test from live camera feed 

Navid Fallahinia - 21/12/2020
BioRobotics Lab

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -x XML_DIR, --xml_dir XML_DIR
                        Path to the folder where the input .xml files are stored.
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

import utils.camerautils as camera

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample inference test from live camera feed ")
parser.add_argument("-dir",
                    "--model_dir",
                    help="Path to the folder where the model is stored.",
                    default='./inference_model/inference_graph_1',
                    type=str)
parser.add_argument("-label",
                    "--label_dir",
                    help="Path to the label map file.",
                    default='./inference_model/labelmap.pbtxt',
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

    PATH_TO_SAVED_MODEL = args.model_dir + "/saved_model"
    PATH_TO_LABELS = args.label_dir

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print('Loading model...', end='')
    start_time = time.time()
    model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # Print PyCapture2 Library Information
    camera.print_build_info()
    # Ensure sufficient cameras are found
    camera.printNumOfCam()
    # Initializing the camera
    camera.init()
    while (True):
        # Capturing image
        camera.capture(model, category_index)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Discconecting the camera
    cv2.destroyAllWindows()
    camera.close()

    input('Done! Press Enter to exit...\n')

""" a wrapper for the Flycapture camera in  python 

Navid Fallahinia - 21/12/2020
BioRobotics Lab
"""

import PyCapture2
import numpy as np
import cv2 
import time

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from utils.inferenceutils import *

# Configurations: 
_PIXEL_FORMAT = PyCapture2.PIXEL_FORMAT.RAW16
_maxWidth = 640
_maxHeight = 480

# Connection to camera
bus = PyCapture2.BusManager()
cam = PyCapture2.Camera()
camInitialised = False

def printNumOfCam():
    """ returns the number of cameras """
    bus = PyCapture2.BusManager()
    numCams = bus.getNumOfCameras()
    if not numCams:
        print('Insufficient number of cameras. Exiting...')
        exit()
    print("Number of cameras detected: ", numCams)
    return numCams

def init(camIndex=0):
    """
        This function initialises the connection with camera starts the capture process
        WARNING: After init() is called, the camera must be properly closed using .close() method
    """
    global camInitialised
    print("Initializing connection to camera ", camIndex)
    cam.connect(bus.getCameraFromIndex( camIndex ))
    __printCameraInfo__(cam)
    # set the format to RAW16 using format 7
    fmt7info, supported = cam.getFormat7Info(0)
    global _PIXEL_FORMAT
    # Check whether pixel format _PIXEL_FORMAT is supported
    if _PIXEL_FORMAT & fmt7info.pixelFormatBitField == 0:
        raise RuntimeError("Pixel format is not supported\n")
    fmt7imgSet = PyCapture2.Format7ImageSettings(0, 0, 0, _maxWidth, _maxHeight, _PIXEL_FORMAT)
    fmt7pktInf, isValid = cam.validateFormat7Settings(fmt7imgSet)
    if not isValid:
        raise RuntimeError("Format7 settings are not valid!")
    cam.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
    cam.setConfiguration(grabMode = PyCapture2.GRAB_MODE.DROP_FRAMES)
    # adjusting camera settings
    autoAdjust()
    setFramerate()
    print_frame_rate()
    print('Starting capture...')
    cam.startCapture()
    camInitialised = True

def capture(model, category_index, display=False):
    """
        This function captures an image and optionally displays the image using openCV.
    """
    if not camInitialised:
        raise RuntimeError("Camera not initialised. Please intialise with init() method")
    try:
        # try retrieving the last image from the camera
        rawImg = cam.retrieveBuffer()
        bgrImg = rawImg.convert(PyCapture2.PIXEL_FORMAT.BGR)
        image_np = np.array(bgrImg.getData(), dtype="uint8").reshape((bgrImg.getRows(), bgrImg.getCols(),3) )
        output_dict = run_inference_for_single_image(model, image_np)
        if display:
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=3)           
            cv2.imshow('frame',image_np)
            cv2.waitKey(10)
        # generating the cropped images
        num_box_images = len([i for i in output_dict['detection_scores'] if i > 0.4])
        box_images = []
        if num_box_images == 0:
            return
        for idx in range(num_box_images):
            box = tuple(output_dict['detection_boxes'][idx].tolist())
            img_cropped = image_crop_single_image(image_np, box)
            box_images.append(img_cropped)
        return box_images

    except PyCapture2.Fc2error as fc2Err:
        print("Error retrieving buffer : ", fc2Err)
        raise RuntimeError("Error retrieving buffer : ", fc2Err)

def close():
    """ This function closes the camera connection and stops image capture"""
    global camInitialised
    camInitialised = False
    print('Disconnecting camera...')
    cam.stopCapture()
    cam.disconnect()

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print('PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3]))
    print()

def print_frame_rate():
    frameRateProp = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
    print('Using frame rate of {}'.format(frameRateProp.absValue))  

def autoAdjust():
    """ Set the camera to auto mode for all settings """
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.BRIGHTNESS, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = True)

def setFramerate(absValue=20):
    """ This function sets the framrate and returns the value from the camera as it will not be exact """
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, autoManualMode = False, absValue = absValue)
    return getFramerate(cam)

def setShutter(absValue = 60):
    """ This function sets the shutter value of the camera to manual mode with the specified value"""
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = False, absValue = absValue)
    return getShutterValue()

def setGain(absValue = 15):
    """ This function sets the gain of the camera to manual mode with a specified value"""
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = False, absValue = absValue)
    return getGainValue()

def setExposure(absValue=1):
    """ This function sets the exposure of the camera to manual mode with a specified value"""
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode = False, absValue = absValue)
    return cam.getProperty( PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE ).absValue

def getFramerate(cam):
    return cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE).absValue

def autoAdjust():
    """ Set the camera to auto mode for all settings """
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = True)
    cam.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, autoManualMode = True)

def getShutterValue():
    """ Returns the current shutter value """
    return cam.getProperty( PyCapture2.PROPERTY_TYPE.SHUTTER ).absValue

def getGainValue():
    """ Returns the current gain value """
    return cam.getProperty( PyCapture2.PROPERTY_TYPE.GAIN ).absValue

def __printCameraInfo__(cam):
    cam_info = cam.getCameraInfo()
    print('\n*** CAMERA INFORMATION ***\n')
    print('Serial number - %d' % cam_info.serialNumber)
    print('Camera model - %s' % cam_info.modelName)
    print('Resolution - %s' % cam_info.sensorResolution)
    print()


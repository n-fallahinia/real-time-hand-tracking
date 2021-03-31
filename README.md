# Real-Time Multi Finger Tracking 
[![alt tag](data/logo.png)](http://robotics.coe.utah.edu/)

_Author: Navid Fallahinia, University of Utah Robotics Center_

_n.fallahinia@utah.edu_

## Requirements

We recommend using python3 and a virtual env. When you're done working on the project, deactivate the virtual environment with `deactivate`.

```
$ virtualenv -p python3 .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

Note that this repository uses Tensorflow 2.3.0 and `tf.keras` API. There are major changes between TF 1 and TF 2 and this program does NOT support TF 1. 

Introduction to TF 1 vs. TF 2:
- [programmer's guide](https://www.tensorflow.org/guide/migrate)

## Task

Given an image of a human finger, returns the bounding box around each finger in the frame using SSD object detection. The model is built based on MobileNet-V2. You can alos read about the SDD here:

- The original paper: [here](https://arxiv.org/abs/1512.02325)
- Some implementation: [here](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

The trained detector must be placed in the `inference_graph` directory. Unfortunately, the finger dataset for training the detector is not publicly available at this time. However, you can email the [author](n.fallahinia@utah.edu) to receive the datatset. 

The tracker model is based on MIL (Multiple Instance Learning) implementation in [here](https://faculty.ucmerced.edu/mhyang/papers/cvpr09a.pdf). The cv::tracker class reference is used for updating the tracker state. The tracker is being called every 10 frames unless there is an unsuccesfull update from the tracker object.

## Quick Run

1. **Tracking from a recorded video**: This is for the case when you want to track the fingers in a video feed. Simply run:

```bash
$ multi_tracking_video_test.py [-v VIDEO_DIR] [-t TRACKER_MDL] [-d SAVE_DIR]
```

2. **Tracking from a live camera feed**: This is the case where you have access to camera and want to track the fingers in real-time from the camera frames. At this point, this code is only capable of working with Flycapture cameras. You can change the camera settings from the `cam_settings.yml` file. 

```
python3 inference_test.py [-h] [-tdir MDL_DIR] [-l LABELS_PATH] [-fdir EST_DIR]

```
**This is a simple demo for this code:**
<p align="center">
    <img src="data/demo.gif" width="250" title="">
</p>


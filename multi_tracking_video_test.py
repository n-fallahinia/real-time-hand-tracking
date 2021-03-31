""" Simple script to run a multi-finger tracking model from a test video 

Navid Fallahinia - 21/12/2020
BioRobotics Lab

usage: multi_tracking_video_test.py [-v VIDEO_DIR] [-t TRACKER_MDL] [-d SAVE_DIR]
"""
import argparse
import imutils
import time

from utils.videoutils import *

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video_input", type=str, default='vid.avi',
	help="Path to input video file")

parser.add_argument("-t", "--tracker", type=str, default="kcf",
	help="Object tracker model")

parser.add_argument("-d", "--video_save", type=bool, default=False,
	help="Path for the output video file")

parser.add_argument("-n", "--object_number", type=int, default=3,
	help="number of fingers in each frame to be found")

# object tracker models. KCF is the default model
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,}

if __name__ == '__main__':
    
	args = parser.parse_args()

	des_object_num = args.object_number
	num_object_added = 0
	frame_size = (600, 853)
	
	# grab the appropriate object tracker using our dictionary of
	trackers = cv2.legacy.MultiTracker_create()
	vs = cv2.VideoCapture(args.video_input)
	image_np = vidCapture(vs, frame_size, verbos=True)

	if args.video_save:
    	# Video recording
		vid_save = vidRecord(frame_size)

	print("Starting the video ...")
	fps = FPS().start()

	# loop over frames from the video stream
	while True:
		image_np = vidCapture(vs, frame_size)
		if image_np is None:
			break

		if num_object_added < des_object_num:
			box = cv2.selectROI("Frame", image_np, fromCenter=False, showCrosshair=True)
			tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
			trackers.add(tracker, image_np, box)
			num_object_added += 1

		success, boxes = trackers.update(image_np)
		# loop over the bounding boxes and draw then on the frame
		if success:
			fps.update()
			fps.stop()					
			image_np = boxProcess(boxes, image_np)

			info = [("Tracker", args.tracker),("FPS", "{:.2f}".format(fps.fps()))]
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(image_np, text, (10, frame_size[1] - ((i * 20) + 20)),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
				
		# show the output frame
		cv2.imshow("Frame", image_np)
		key = cv2.waitKey(20) & 0xFF

		if num_object_added >= des_object_num:
			if args.video_save:
				vid_save.write(image_np)
			# box_images = []
			for box_idx, box in enumerate(boxes):
				img_cropped = video_crop_from_frame(image_np, box)
				cv2.imshow("finger {:2d}".format(box_idx), img_cropped)

		if key == ord("q"):
			break
	vs.release()
	cv2.destroyAllWindows()
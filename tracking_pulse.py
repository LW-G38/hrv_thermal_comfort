# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from matplotlib import pyplot as plt
import io
import pandas as pd

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# face recognition coefficients
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5

# file name that you are saving the data to
table_name = "hrv_wei.xlsx"

# heartbeat setup
heartbeat_count = 128
heartbeat_values = [0]*heartbeat_count
heartbeat_times = [time.time()]*heartbeat_count

# Matplotlib graph surface
fig = plt.figure()
ax = fig.add_subplot(111)
heart = []

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			# cv2.rectangle(frame, (startX, startY), (endX, endY),
			# 	(0, 255, 0), 2)

			# get heartbeat reading area - left eye
			# TODO: you can adjust it based on your face
			startXhrv = int(startX + (endX - startX)/3)
			endXhrv = int(endX - (endX - startX)/8)
			startYhrv = int(startY + (endY - startY)/4)
			endYhrv = int(endY - (endY - startY)/2)
			cv2.rectangle(frame, (startXhrv, startYhrv), (endXhrv, endYhrv),
				(0, 255, 0), 2)
			
			#Prepare for heartbeat update
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			crop_img = img[startYhrv:endYhrv, startXhrv:endXhrv]

			# Update heartbeat data
			heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
			heartbeat_times = heartbeat_times[1:] + [time.time()]
			heart.append(heartbeat_values)

    		# Draw matplotlib graph to numpy array
			ax.plot(heartbeat_times, heartbeat_values)
			fig.canvas.draw()
			plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),
				dtype=np.uint8, sep='')
			ncols, nrows = fig.canvas.get_width_height()
			plot_img_np = plot_img_np.reshape(nrows*2, ncols*2, 3)
			plt.cla()

			cv2.imshow('Graph', plot_img_np)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	# for (objectID, centroid) in objects.items():
	# 	# draw both the ID of the object and the centroid of the
	# 	# object on the output frame
	# 	text = "ID {}".format(objectID)
	# 	cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# 	cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# stop the program if we are having another face
	# since we are only reading one person
	if len(objects) > 1:
		heart_df = pd.DataFrame(heart)
		heart_df.to_excel(table_name)
		break

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		heart_df = pd.DataFrame(heart)
		heart_df.to_excel(table_name)
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
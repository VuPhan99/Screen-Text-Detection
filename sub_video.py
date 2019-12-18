# USAGE
# python text_detection_video.py --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
import concurrent.futures
from decode_predictions import decode_predictions

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	
	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1280)
	orig = frame.copy()

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	# construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes


	#__Phoenix get rects, confidences from threading
	with concurrent.futures.ThreadPoolExecutor() as executor:
		future = executor.submit(decode_predictions, scores, geometry, args)
		(rects, confidences) = future.result()

	# (rects, confidences) = decode_predictions(scores, geometry) #old function
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the frame
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		#extract the region of interest
		r = orig[startY:endY, startX:endX]
		
		#configuration setting to convert image to string.  
		configuration = ("-l eng --oem 1 --psm 8")
		##This will recognize the text from the image of bounding box
		text = pytesseract.image_to_string(r, config=configuration)
		#Ideas: sau khi ve hinh vuong detect thi write below duoi cai detect do'
		print(text)
		cv2.putText(orig, text, (startX, startY + 100),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
		
		# show the output image
		# cv2.imshow("Text Detection", output)
		# cv2.waitKey(10)

	# update the FPS counter
	fps.update()
	
	
	# show the output frame
	# orig = cv2.putText(frame, 'OpenCV', (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,  
    #                1, (255, 0, 0) , 1, cv2.LINE_AA) 
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(10) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
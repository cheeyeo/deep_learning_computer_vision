# Simple opencv face detection using built-in cascade classifier model
# Here we adjust the scale factor and min-neigbours arguments passed to detectMultiScale to try and improve the false bounding boxes detected...

from cv2 import imread
from cv2 import CascadeClassifier
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import rectangle

pixels = imread('test2.jpg')

# Load pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# Perform face detection
bboxes = classifier.detectMultiScale(pixels, 1.005, 6)

for box in bboxes:
	x, y, width, height = box
	x2, y2 = x + width, y + height
	rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)

imshow('face detection', pixels)

# Keep window open until we press a key
waitKey(0)

# Close the window
destroyAllWindows()


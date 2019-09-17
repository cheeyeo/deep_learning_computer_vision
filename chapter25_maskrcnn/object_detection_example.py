# Example on object detection using Mask R-CNN Library
# Uses a pre-trained Mask R-CNN model trained on MSCOCO dataset

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from mrcnn.visualize import display_instances
from mrcnn.model import MaskRCNN
import os
import argparse
from utils import draw_image_with_boxes, load_coco_classes
from config import TestConfig

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Image to perform object recognition on.")
ap.add_argument("-m", "--model", default="data/mask_rcnn_coco.h5", type=str, help="Model weights for Mask R-CNN model.")
ap.add_argument("-o", "--object-detection", action="store_true", help="Perform object detection using Mask R-CNN model.")
args = vars(ap.parse_args())

# Define and load model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

rcnn.load_weights(args["model"], by_name=True)

img = load_img(args["image"])
img_pixels = img_to_array(img)

results = rcnn.detect([img_pixels], verbose=0)
r = results[0]

if args["object_detection"]:
	print("[INFO] Performing object detection using display_instances...")
	
	# define 81 classes that the coco model knowns about
	class_names = load_coco_classes('data/coco_classes.txt')

	display_instances(img_pixels, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
else:
	draw_image_with_boxes(img, r['rois'])
	print('[INFO] Saving image with bounding boxes')
	img.save(os.path.join('out', args["image"]))
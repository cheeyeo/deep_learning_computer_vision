from PIL import Image, ImageDraw
import numpy as np

def load_coco_classes(filename):
	with open(filename, 'r') as f:
		classes = f.readlines()
	classes = [c.strip() for c in classes]
	return classes

def draw_image_with_boxes(image, boxes):
	thickness = (image.size[0] + image.size[1]) // 300

	draw = ImageDraw.Draw(image)

	for box in boxes:
		top, left, bottom, right = box
		top = max(0, np.floor(top + 0.5).astype('int32'))
		left = max(0, np.floor(left + 0.5).astype('int32'))
		bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
		right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

		for i in range(thickness):
			draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 0, 0))
		del draw

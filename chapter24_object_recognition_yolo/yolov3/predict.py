from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Image file to predict on.")
ap.add_argument("-m", "--model", type=str, default="yolov3.h5", help="Model file to load")
args = vars(ap.parse_args())

def load_image_pixels(img, shape):
	image = load_img(img)
	width, height = image.size
	# load image with required size
	image = load_img(img, target_size=shape)
	image = img_to_array(image)
	image = image.astype('float32')
	image /= 255.0
	image = np.expand_dims(image, 0)
	return image, width, height

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

# Apply Non-maximal suppression
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# After applying NMS, we filter out the bounding boxes
# that indicate strong presence of an object that are
# above threshold value
def get_boxes(boxes, labels, thresh):
	vboxes, vlabels, vscores = list(), list(), list()

	for box in boxes:
		for i in range(len(labels)):
			if box.classes[i] > thresh:
				vboxes.append(box)
				vlabels.append(labels[i])
				vscores.append(box.classes[i]*100)

	return vboxes, vlabels, vscores

def draw_boxes(image, out_scores, out_boxes, out_classes):
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = c
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label)

        top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print('[INFO] Prediction: {}, Bounding boxes: {}, {}'.format(label, (left, top), (right, bottom)))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 255, 255))
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=None)
        del draw

def read_coco_classes(filename):
    with open('data/coco_classes.txt', 'r') as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes

model = load_model(args['model'])
# model.summary()

input_w, input_h = 416, 416

img_name = args['image']
image = Image.open(img_name)
image_data, image_w, image_h = load_image_pixels(img_name, (input_w, input_h))

yhat = model.predict(image_data)
print([a.shape for a in yhat])

anchors = [[116,90, 156,198, 373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
	boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_w, input_h)

correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

# Non-maximal suppression
# Removes overlapping bounding boxes
do_nms(boxes, 0.5)

labels = read_coco_classes('data/coco_classes.txt')

# Performs score thresholding? Remove boxes that have predicted a class less than threshold
vboxes, vlabels, vscores = get_boxes(boxes, labels, class_threshold)
for i in range(len(vboxes)):
	print('[INFO] Predicted {} with score {:.2f}'.format(vlabels[i], vscores[i]))

draw_boxes(image, vscores, vboxes, vlabels)

print('[INFO] Saving image with bounding boxes and probs ....')
image.save(os.path.join('out', img_name))
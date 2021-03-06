import numpy as np
import tensorflow as tf
from keras import backend as K
from yolo_utils import scale_boxes
from yad2k.models.keras_yolo import yolo_boxes_to_corners

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
  """
	Filters YOLO boxes by thresholding on object and class confidence.
  Arguments:
  box_confidence -- tensor of shape (19, 19, 5, 1)
  boxes -- tensor of shape (19, 19, 5, 4)
  box_class_probs -- tensor of shape (19, 19, 5, 80)
  threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
  
  Returns:
  scores -- tensor of shape (None,), containing the class probability score for selected boxes
  boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
  classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
  
  Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
  For example, the actual output size of scores would be (10,) if there are 10 boxes.
  """

  # Step 1: Compute box scores
  box_scores = box_confidence * box_class_probs

  # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
  box_classes = K.argmax(box_scores, axis=-1)
  box_class_scores = K.max(box_scores, axis=-1)

  # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
  # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
  filtering_mask = box_class_scores >= threshold

  # Step 4: Apply the mask to scores, boxes and classes
  scores = tf.boolean_mask(box_class_scores, filtering_mask)
  boxes = tf.boolean_mask(boxes, filtering_mask)
  classes = tf.boolean_mask(box_classes, filtering_mask)

  return scores, boxes, classes

def iou(box1, box2):
	"""
	Implement the intersection over union (IoU) between box1 and box2

  Arguments:
  box1 -- first box, list object with coordinates (x1, y1, x2, y2)
  box2 -- second box, list object with coordinates (x1, y1, x2, y2)
	"""
	xi1 = np.maximum(box1[0], box2[0])
	yi1 = np.maximum(box1[1], box2[1])
	xi2 = np.minimum(box1[2], box2[2])
	yi2 = np.minimum(box1[3], box2[3])
	inter_area = np.max((yi2 - yi1), 0) * np.max((xi2 - xi1), 0)
	box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
	box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
	union_area = box1_area + box2_area - inter_area
	iou = inter_area / union_area
	return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
  """
  Applies Non-max suppression (NMS) to set of boxes
  
  Arguments:
  scores -- tensor of shape (None,), output of yolo_filter_boxes()
  boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
  classes -- tensor of shape (None,), output of yolo_filter_boxes()
  max_boxes -- integer, maximum number of predicted boxes you'd like
  iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
  
  Returns:
  scores -- tensor of shape (, None), predicted score for each box
  boxes -- tensor of shape (4, None), predicted box coordinates
  classes -- tensor of shape (, None), predicted class for each box
  
  Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
  function will transpose the shapes of scores, boxes, classes. This is made for convenience.
  """

  # tensor to be used in tf.image.non_max_suppression()
  max_boxes_tensor = K.variable(max_boxes, dtype='int32')

  # initialize variable max_boxes_tensor
  K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

  # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
  nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

  # Use K.gather() to select only nms_indices from scores, boxes and classes
  scores = tf.gather(scores, nms_indices)
  boxes = tf.gather(boxes, nms_indices)
  classes = tf.gather(classes, nms_indices)

  return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
  """
	Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

	Arguments:
  yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
  box_confidence: tensor of shape (None, 19, 19, 5, 1)
  box_xy: tensor of shape (None, 19, 19, 5, 2)
  box_wh: tensor of shape (None, 19, 19, 5, 2)
  box_class_probs: tensor of shape (None, 19, 19, 5, 80)
  image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
  max_boxes -- integer, maximum number of predicted boxes you'd like
   score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
  iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

  Returns:
  scores -- tensor of shape (None, ), predicted score for each box
  boxes -- tensor of shape (None, 4), predicted box coordinates
  classes -- tensor of shape (None,), predicted class for each box
	"""

  # Retrieve outputs of the YOLO model
  box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

  # Convert boxes to be ready for filtering functions 
  boxes = yolo_boxes_to_corners(box_xy, box_wh)

  # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold
  scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

  # Scale boxes back to original image shape.
  boxes = scale_boxes(boxes, image_shape)

  scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

  return scores, boxes, classes


if __name__ == "__main__":
	print('Testing yolo_eval ...')
	with tf.Session() as test_b:
		yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))

		scores, boxes, classes = yolo_eval(yolo_outputs)
		print("scores[2] = " + str(scores[2].eval()))
		print("boxes[2] = " + str(boxes[2].eval()))
		print("classes[2] = " + str(classes[2].eval()))
		print("scores.shape = " + str(scores.eval().shape))
		print("boxes.shape = " + str(boxes.eval().shape))
		print("classes.shape = " + str(classes.eval().shape))

  # Expected Output:
  # scores[2] 138.791
  # boxes[2]  [ 1292.32971191 -278.52166748 3876.98925781 -835.56494141]
  # classes[2]  54
  # scores.shape  (10,)
  # boxes.shape (10, 4)
  # classes.shape (10,)


	# print('Testing yolo_filter_boxes...')
	# with tf.Session() as test_a:
	# 	box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
	# 	boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
	# 	box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
	# 	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
	# 	print("scores[2] = " + str(scores[2].eval()))
	# 	print("boxes[2] = " + str(boxes[2].eval()))
	# 	print("classes[2] = " + str(classes[2].eval()))
	# 	print("scores.shape = " + str(scores.shape))
	# 	print("boxes.shape = " + str(boxes.shape))
	# 	print("classes.shape = " + str(classes.shape))

	# Expected output:
	# scores[2]	10.7506
	# boxes[2]	[ 8.42653275 3.27136683 -0.5313437 -4.94137383]
	# classes[2]	7
	# scores.shape	(?,)
	# boxes.shape	(?, 4)
	# classes.shape	(?,)


	# print('Testing iou function...')
	# box1 = (2, 1, 4, 3)
	# box2 = (1, 2, 3, 4) 
	# print("iou = " + str(iou(box1, box2)))

	# iou = 0.14285714285714285


	# print('Testing yolo_non_max_suppression...')
	# with tf.Session() as test_b:
	# 	scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
	# 	boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
	# 	classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
	# 	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
	# 	print("scores[2] = " + str(scores[2].eval()))
	# 	print("boxes[2] = " + str(boxes[2].eval()))
	# 	print("classes[2] = " + str(classes[2].eval()))
	# 	print("scores.shape = " + str(scores.eval().shape))
	# 	print("boxes.shape = " + str(boxes.eval().shape))
	# 	print("classes.shape = " + str(classes.eval().shape))

	# Expected output:
  # scores[2] 6.9384
  # boxes[2]  [-5.299932 3.13798141 4.45036697 0.95942086]
  # classes[2]  -2.24527
  # scores.shape  (10,)
  # boxes.shape (10, 4)
  # classes.shape (10,)
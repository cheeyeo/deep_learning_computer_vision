import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imread
import scipy.io
import scipy.misc
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model
from yolo_functions import yolo_eval
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head

def predict(sess, image_file):
  """
  Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

  Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
  """

  # Preprocess your image
  image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

  # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
  out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

  # Print predictions info
  print('Found {} boxes for {}'.format(len(out_boxes), image_file))

  # Generate colors for drawing bounding boxes.
  colors = generate_colors(class_names)

  # Draw bounding boxes on the image file
  draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

  # Save the predicted bounding box on the image
  image.save(os.path.join("out", image_file), quality=90)

  # Display the results in the notebook
  output_image = imread(os.path.join("out", image_file))
  imshow(output_image)

  return out_scores, out_boxes, out_classes

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)   

yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()

"""
yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output
yolo_model.output is processed by yolo_head. It gives you yolo_outputs
yolo_outputs goes through a filtering function, yolo_eval. It outputs your predictions: scores, boxes, classes
"""

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
"""
Found 7 boxes for test.jpg
car 0.60 (925, 285) (1045, 374)
car 0.66 (706, 279) (786, 350)
bus 0.67 (5, 266) (220, 407)
car 0.70 (947, 324) (1280, 705)
car 0.74 (159, 303) (346, 440)
car 0.80 (761, 282) (942, 412)
car 0.89 (367, 300) (745, 648)
"""
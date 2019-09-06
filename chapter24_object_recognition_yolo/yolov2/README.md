## Object Detection with YOLOv2

Recreate example from jupyter notebook of Coursera week 3 course on Sequence Models

Create a YOLOv2 model using code from project https://github.com/allanzelener/YAD2K

## Notes
* ```yolo_eval()``` which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which we have provided):
```
boxes = yolo_boxes_to_corners(box_xy, box_wh)
```
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes

```
boxes = scale_boxes(boxes, image_shape)
```

YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image--for example, the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.


* Input image (608, 608, 3)
* The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
* After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
* Each cell in a 19x19 grid over the input image gives 425 numbers.
* 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
* 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and and 80 is the number of classes we'd like to detect

You then select only few boxes based on:
Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
This gives you YOLO's final output.



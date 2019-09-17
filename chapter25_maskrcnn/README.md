# Object Detection with Mask R-CNN

Exercises from chapter 25 of Deep Learning for Computer Vision

Build a simple Mask R-CNN model using pre-trained model from following repo:
https://github.com/matterport/Mask_RCNN

Need to download the sample weights from the repo above before building model

To run:
```
python object_detection_example.py -i <name of image>
```

Above will detect and draw a bounding box around detected object and save it into `out` sub directory

To run the Mask R-CNN library version of object recognition:
```
python object_detection_example.py -i <name of image> -o
```

This will attempt to draw bounding boxes and masks using matplotlib hence may not work on Linux as matplotlib uses `agg`
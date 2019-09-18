# Develop Custom Object Detection Model

Chapter 26 from Deep Learning For Computer Vision

Build a custom object detection model using the Mask R-CNN library to recognise kangaroos from custom dataset.

The custom dataset can be cloned into the local directory from here:
https://github.com/experiencor/kangaroo

You also need to download the model's weights here:
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5


To perform training on the model:
```
python train.py
```

To perform evaluation on the model:
```
python evaluate.py
```

To perform prediction on the model:
```
python predict.py
```


## References

* [MatterPort Mask RCNN](https://github.com/matterport/Mask_RCNN)


## Extensions

* Experiment by tuning the model for better performance, such as using a different learning rate

* Apply model to detect kangaroos on new photos

* Find or develop a small dataset for object detection and develop a mask R CNN model for it

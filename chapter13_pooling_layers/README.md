## How Pooling layers work

Chapter 13 of book Deep Learning for Computer Vision


## Extensions

* Develop a small model to demonstrate global max pooling

* Experiment with different stride sizes with a pooling layer and its effect on model summary

* Experiment with different padding sizes with a pooling layer and its effect on model sumary


## Notes on Pooling Layer

* Limitation of CNN feature maps is that it records the precise location of features in the input; small movements in the position of the feature in the input image will result in different feature map

can happen with re-cropping, scaling, rotation to input image

i.e. when using ImageDataGenerator in keras to apply image data augmentation


* Pooling downsamples the feature map to a smaller size
  e.g. pooling layer applied with filter size of 2x2, stride 2, which halves the size of the original feature map.

  e.g. 6x6 => 3x3

 * Pooling layer added after non-linearity e.g. ReLU layer


 Input image

 			||

 Convolutional Layer

 			||

 Non linearity

 			||

 Pooling layer

* Pooling layer has a default filter size of (2,2) and stride of (2,2)


* 3 main types of pooling layers in Keras:
	* Average Pooling
	* Max pooling
	* Global pooling

* Average pooling takes the average of each patch of feature map; downsamples the feature map say from 6x6 to 3x3

* Max pooling calculates maximum or largest feature in each patch of feature map.

  This is more accurate as feature maps encode spatial presence of some form of pattern over the different tiles of feature maps; hence more informative to look at maximum presence than average presence of some features


* Global pooling downsamples the entire feature map to a single value

	Summarize presence of feature in an image

	Transform feature map to an output prediction for model

	Sometimes used to transition from feature maps to output prediction for the model

* 2 types of global pooling layers:

  * GlobalAveragePooling2D
  * GlobalMaxPooling2D
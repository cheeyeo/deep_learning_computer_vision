## How Pooling layers work

Chapter 13 of book Deep Learning for Computer Vision


## Extensions

* Develop a small model to demonstrate global max pooling

* Experiment with different stride sizes with a pooling layer and its effect on model summary

* Experiment with different padding sizes with a pooling layer and its effect on model sumary


## Notes on Pooling Layer

* Limitation of CNN feature maps is that it records the precise location of features in the input; small movements in the position of the feature in the input image will result in different feature map

can happen with re-cropping, scaling, rotation to input image

i.e. when using imagedatagenerator in keras to apply image data augmentation


* Pooling downsamples the feature map to a smaller size
  e.g. pooling layer applied with filter size of 2x2, stride 2, which halves th size of the original feature map.

  e.g. 6x6 => 3x3

 * Pooling layer added after non-linearity e.g. ReLU layer


 Input image

 			||

 Convolutional Layer

 			||

 Non linearity

 			||

 Pooling layer



* Global pooling downsamples the entire feature map to a single value

	Summarize presence of feature in an image

	Transform feature map to an output prediction for model
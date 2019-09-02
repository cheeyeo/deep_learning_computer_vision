## Image Channel Ordering

Exercises from chapter10 of `Deep Learning For Computer Vision`

## Channel Ordering Notes

Channel First => [channel][rows][cols]

Channel Last => [rows][cols][channel]

* Query current channel ordering:
```
import keras.backend as K

print(K.image_data_format())
```

## Extensions

* Change default channel ordering on local workstation and confirms that it works

  Change `image_data_format` value in `~/.keras/keras.json` and confirm that the framework picks it up

* Develop a small CNN network that uses channels_first and channels_last ordering and review model summary

	Code in `cnn_model.py`

	For channel_last ordering, which is the default:
	```
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 28, 28, 1)         0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 26, 26, 2)         20        
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 13, 13, 2)         0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 338)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 10)                3390      
	_________________________________________________________________
	dense_2 (Dense)              (None, 1)                 11        
	=================================================================
	Total params: 3,421
	Trainable params: 3,421
	Non-trainable params: 0
	```

	The output from the conv2d to max pooling layers have the channel as the last value

	For channel_first ordering, by setting `image_data_format` in `~/.keras/keras.json`:

	```
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 1, 28, 28)         0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 2, 26, 26)         20        
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 2, 13, 13)         0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 338)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 10)                3390      
	_________________________________________________________________
	dense_2 (Dense)              (None, 1)                 11        
	=================================================================
	Total params: 3,421
	Trainable params: 3,421
	Non-trainable params: 0
	```

	The channel value is set to the first value for the inputs, conv2d and max pooling layers.


* Develop a function that will change the structure of an image dataset based on configured channel ordering

	Code in `dataset_channel_ordering.py`
# Example of creating CNN model with VGG blocks
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

def vgg_block(layer_in, num_filters, num_conv):
	for _ in range(num_conv):
		layer_in = Conv2D(num_filters, (3,3), padding='same', activation='relu')(layer_in)
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

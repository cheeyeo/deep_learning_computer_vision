from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import add

# Implementation of 1 ResNet block
def residual_block(layer_in, num_filters):
	merge_input = layer_in
	# if num_filters don't match the num of filters in the input layer
	# we use a 1x1 conv to reshape the num of filters
	if layer_in.shape[-1] != num_filters:
		merge_input = Conv2D(num_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

	# conv1
	conv1 = Conv2D(num_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

	# conv2
	conv2 = Conv2D(num_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)

	layer_out = add([conv2, merge_input])
	layer_out = Activation('relu')(layer_out)
	return layer_out
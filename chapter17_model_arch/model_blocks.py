from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Activation

def vgg_block(layer_in, num_filters, num_conv):
	"""
	Creates a VGG block parameterized by number of filters
	and number of layers
	"""

	for _ in range(num_conv):
		layer_in = Conv2D(num_filters, (3,3), padding="same", activation="relu")(layer_in)

	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in


def inception_module(layer_in, f1, f2, f3):
	"""
	Block of parallel conv layers with different filter sizes
	(1x1, 3x3, 5x5), followed by a maxpooling layer of (3,3)
	and concatenate the results

	f1 - Number of filters for 1x1 conv layer
	f2 - Number of filters for 3x3 conv layer
	f3 - Number of filters for 5x5 conv layer
	"""

	conv1 = Conv2D(f1, (1,1), padding="same", activation="relu")(layer_in)

	conv2 = Conv2D(f2, (3,3), padding="same", activation="relu")(layer_in)

	conv3 = Conv2D(f3, (5,5), padding="same", activation="relu")(layer_in)

	pool = MaxPooling2D((3,3), strides=(1,1), padding="same")(layer_in)

	layer_out = concatenate([conv1, conv2, conv3, pool], axis=-1)

	return layer_out

def optimized_inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	"""
	Optimized version of inception module which uses 1x1 conv layers before 3x3 and 5x5 layers
	"""

	conv1 = Conv2D(f1, (1,1), padding="same", activation="relu")(layer_in)

	conv3 = Conv2D(f2_in, (1,1), padding="same", activation="relu")(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding="same", activation="relu")(conv3)

	conv5 = Conv2D(f3_in, (1,1), padding="same", activation="relu")(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding="same", activation="relu")(conv5)

	pool = MaxPooling2D((3,3), strides=(1,1), padding="same")(layer_in)
	pool = Conv2D(f4_out, (1,1), padding="same", activation="relu")(pool)

	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)

	return layer_out

def residual_module(layer_in, num_filters):
	"""
	Implementation of the residual identity module
	for ResNet
	"""

	merge_input = layer_in

	if layer_in.shape[-1] != num_filters:
		merge_input = Conv2D(num_filters, (1,1), padding="same", activation="relu", kernel_initializer="he_normal")(layer_in)

	conv1 = Conv2D(num_filters, (3,3), padding="same", activation="relu", kernel_initializer="he_normal")(layer_in)

	conv2 = Conv2D(num_filters, (3,3), padding="same", activation="linear", kernel_initializer="he_normal")(conv1)

	# add filters
	layer_out = add([conv2, merge_input])

	layer_out = Activation("relu")(layer_out)

	return layer_out


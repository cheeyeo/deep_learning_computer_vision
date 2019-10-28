# Example of using multiple VGG blocks where the nos of filters increase with depth of model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from model_blocks import vgg_block


input = Input(shape=(256, 256, 3))

# Create 2 conv layers with 64 filters
layer = vgg_block(input, 64, 2)

# Create 2 conv layers with 128 filters
layer = vgg_block(layer, 128, 2)

# Create 4 conv layers with 256 filters
layer = vgg_block(layer, 256, 4)

model = Model(inputs=input, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file="artifacts/vgg_complex.png")
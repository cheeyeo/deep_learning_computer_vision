# Example usage of ResNet identity module

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from model_blocks import residual_module
from tensorflow.keras.utils import plot_model

input = Input(shape=(256, 256, 3))

layer = residual_module(input, 64)

model = Model(inputs=input, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file="residual_model.png")
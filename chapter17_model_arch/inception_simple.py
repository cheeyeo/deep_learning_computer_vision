# Example of using a single Inception block

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from model_blocks import inception_module

input = Input(shape=(256, 256, 3))

layer = inception_module(input, 64, 128, 32)

model = Model(inputs=input, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file="artifacts/simple_inception_model.png")
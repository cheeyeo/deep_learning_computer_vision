from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from model_blocks import optimized_inception_module

input = Input(shape=(256, 256, 3))

# Add inception block
layer = optimized_inception_module(input, 64, 96, 128, 16, 32, 32)

# Add inception block
layer = optimized_inception_module(layer, 128, 128, 192, 32, 96, 64)

model = Model(inputs=input, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file="artifacts/multi_inception_model.png")
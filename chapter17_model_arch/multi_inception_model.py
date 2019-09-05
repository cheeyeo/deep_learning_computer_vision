from inception_block import projected_inception_module as inception_module
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

img_data = Input(shape=(256, 256, 3))

layer = inception_module(img_data, 64, 96, 128, 16, 32, 32)

layer = inception_module(layer, 128, 128, 192, 32, 96, 64)

model = Model(inputs=img_data, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file='multi_inception_model.png')
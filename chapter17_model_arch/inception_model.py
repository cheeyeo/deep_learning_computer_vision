from inception_block import inception_module as inception_module
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

img_data = Input(shape=(256, 256, 3))
layer = inception_module(img_data, 64, 128, 32)
model = Model(inputs=img_data, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file='artifacts/simple_inception_model.png')
from residual_module import residual_block as residual_block
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

img_data = Input(shape=(256, 256, 3))

layer = residual_block(img_data, 64)

model = Model(inputs=img_data, outputs=layer)
model.summary()

plot_model(model, show_shapes=True, to_file='artifacts/residual_model.png')
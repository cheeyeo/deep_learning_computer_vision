from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from vgg_module import vgg_block as vgg_block

img_input = Input(shape=(256, 256, 3))
vgg_blocks = vgg_block(img_input, 64, 2)
model = Model(inputs=img_input, outputs=vgg_blocks)
model.summary()
plot_model(model, show_shapes=True, to_file='artifacts/vgg_block.png')
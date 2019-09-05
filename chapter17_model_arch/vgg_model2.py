from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from vgg_module import vgg_block as vgg_block

img_input = Input(shape=(256, 256, 3))

# add vgg block
layer = vgg_block(img_input, 64, 2)

# add vgg block
layer = vgg_block(layer, 128, 2)

# add vgg block
layer = vgg_block(layer, 256, 4)

model = Model(inputs=img_input, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file='artifacts/multiple_vgg_blocks.png')
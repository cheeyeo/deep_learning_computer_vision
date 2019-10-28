# Example implementing a single VGG block
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from model_blocks import vgg_block


input = Input(shape=(256, 256, 3))

layer = vgg_block(input, 64, 2)

model = Model(inputs=input, outputs=layer)

model.summary()

plot_model(model, show_shapes=True, to_file="artifacts/vgg_simple.png")
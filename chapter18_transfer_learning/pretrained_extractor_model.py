# Example of using VGG16 model via fine tuning

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


model = VGG16(weights="imagenet", include_top=False, input_shape=(300, 300, 3))

# Add new classifier layers
flat1 = Flatten()(model.outputs[0])
class1 = Dense(1024, activation="relu")(flat1)
output = Dense(10, activation="softmax")(class1)

model = Model(inputs=model.inputs, outputs=output)

model.summary()
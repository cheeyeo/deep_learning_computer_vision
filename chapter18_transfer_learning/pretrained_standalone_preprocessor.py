# Example of using VGG16 model as a standalone feature preprocessor

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

img = load_img("dog.jpg", target_size=(224, 224))

img = img_to_array(img)

img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

img = preprocess_input(img)

model = VGG16()
# Note: model.layers.pop() does not remove the last output layer; need to use model._layers for tf.keras...
# REF: https://github.com/tensorflow/tensorflow/issues/22479
# https://github.com/keras-team/keras/issues/8909
model._layers.pop()

model2 = Model(inputs=model.input, outputs=model.layers[-1].output)
model2.summary()

features = model2.predict(img)
# # # print(features)
print("[INFO] Extracted features: {}".format(features.shape))
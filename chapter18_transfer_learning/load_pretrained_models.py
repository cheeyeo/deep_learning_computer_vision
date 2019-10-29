# Example of loading pre-trained models

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

print("[INFO] Loading VGG16 model...")
model = VGG16()
model.summary()

print("[INFO] Loading InceptionV3 model...")
model = InceptionV3()
model.summary()

print("[INFO] Loading ResNet50 model...")
model = ResNet50()
model.summary()
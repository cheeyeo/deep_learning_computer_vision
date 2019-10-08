# Example of loading a VGG Face2 model
# keras_vggface provides 3 pre-trained VGGFace models:
# VGGFace(model='vgg16') => VGGFace model
# VGGFace(model='resnet50') => VGGFace2 model
# VGGFace(model='senet50') => VGGFace2 model

from keras_vggface.vggface import VGGFace

model = VGGFace(model='resnet50')

print('Input shape: {}'.format(model.inputs))
print('Output shape: {}'.format(model.outputs))
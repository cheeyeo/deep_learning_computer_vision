# Example of simple face identification using VGG Face model
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", help="Path to photo.")
args = vars(ap.parse_args())

# Extract single face with given photo
def extract_face(filename, required_size=(224, 224)):
	image = Image.open(filename)
	pixels = np.asarray(image)

	detector = MTCNN()
	results = detector.detect_faces(pixels)

	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height

	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)

	face_array = np.asarray(image)
	return face_array

pixels = extract_face(args['filename'])
# plt.imshow(pixels)
# plt.savefig('test.jpg')
print(type(pixels))

pixels = pixels.astype('float32')
samples = np.expand_dims(pixels, axis=0)

# Preprocess input same way as training data i.e. centering mean pixel values
samples = preprocess_input(samples, version=2)

# create VGGFace model
model = VGGFace(model='resnet50')

yhat = model.predict(samples)

results = decode_predictions(yhat)

for result in results[0]:
	print('{}, {:.3f}'.format(result[0][3:-1], result[1] * 100))
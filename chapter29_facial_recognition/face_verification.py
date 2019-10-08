# Example of simple face verification using VGG Face model
"""
For face verification, we calculate a face embedding for a given new face and compare it with the embedding of a face known to the system.

Face embedding is a vector representing features extracted from a face.
"""
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import scipy.spatial

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

def get_embeddings(filenames):
	faces = [extract_face(f) for f in filenames]

	samples = np.asarray(faces, 'float32')

	samples = preprocess_input(samples, version=2)

	# To retrieve the face embeddings, we need to exclude the classifier from the model and set pooling to avg so that filter maps at end of output are reduced to vector using global avg pooling
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

	yhat = model.predict(samples)

	return yhat

# Determine if candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, threshold=0.5):
	"""
	Use cosine similarity to calculate the distance between known and
	candidate embeddings.

	Similar embeddings will have a low cosine distance whereas mismatched embeddings will have a value greater than threshold
	"""

	# Calculate distance between embeddings
	score = scipy.spatial.distance.cosine(known_embedding, candidate_embedding)

	if score <= threshold:
		print('[INFO] Face is a match: {:.3f} <= {:.3f}'.format(score, threshold))
	else:
		print('[INFO] Face is NOT a match: {:.3f} > {:.3f}'.format(score, threshold))


filenames = ['sharon_stone1.jpg', 'sharon_stone2.jpg', 'sharon_stone3.jpg', 'channing_tatum.jpg']

embeddings = get_embeddings(filenames)

known_id = embeddings[0]

print('Positive Tests')
is_match(known_id, embeddings[1])
is_match(known_id, embeddings[2])

print('Negative Tests')
is_match(known_id, embeddings[3])
import numpy as np
from keras.models import load_model

def get_embedding(model, face_pixels):
	"""
	Gets the face embedding for a given image
	using the output of the Facenet model
	"""

	# Standardize pixel values as required by Facenet
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across all channels
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	samples = np.expand_dims(face_pixels, axis=0)

	yhat = model.predict(samples)

	return yhat[0]

# Load face dataset
data = np.load('faces-dataset.npz')

trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print('[INFO] Loaded dataset shape: trainX: {}, trainY: {}, testX: {}, testY: {}'.format(trainX.shape, trainY.shape, testX.shape, testY.shape))

model = load_model('facenet_keras.h5')
print('[INFO] Loaded model')
# model.summary()

# Convert each face in train set into embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	print(embedding.shape)
	newTrainX.append(embedding)

newTrainX = np.asarray(newTrainX)
print('[INFO] newTrainX shape: {}'.format(newTrainX.shape))

# Convert each face in test set into embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)

newTestX = np.asarray(newTestX)
print('[INFO] newTestX shape: {}'.format(newTestX.shape))

np.savez_compressed('faces-embeddings.npz', newTrainX, trainY, newTestX, testY)

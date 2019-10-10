from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)

	image = image.convert('RGB')

	pixels = np.asarray(image)

	detector = MTCNN()

	results = detector.detect_faces(pixels)

	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	# extract face
	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)

	face_array = np.asarray(image)
	return face_array


def load_faces(directory):
	faces = list()

	for filename in os.listdir(directory):
		filename = os.path.sep.join([directory, filename])
		face = extract_face(filename)
		faces.append(face)

	return faces

def load_dataset(directory):
	X, y = list(), list()

	# Enumerate subfolders
	for subdir in os.listdir(directory):
		path = os.path.sep.join([directory, subdir])
		print('[INFO] Path is: {}'.format(path))

		if not os.path.isdir(path):
			continue

		faces = load_faces(path)

		labels = [subdir for _ in range(len(faces))]

		print('[INFO] Loaded {:d} examples for class {}'.format(len(faces), subdir))

		X.extend(faces)
		y.extend(labels)

	return np.asarray(X), np.asarray(y)



if __name__ == '__main__':
	celeb_name = 'ben_afflek'
	folder = 'train'
	current_dir = os.getcwd()
	current_dir = os.path.sep.join([current_dir, folder, celeb_name])
	i = 1

	for filename in os.listdir(current_dir):
		path = os.path.sep.join([current_dir, filename])
		print('PATH: ', path)
		face = extract_face(path)
		print(i, face.shape)

		plt.subplot(2, 7, i)
		plt.axis('off')
		plt.imshow(face)
		i += 1

	plt.savefig('test.jpg')

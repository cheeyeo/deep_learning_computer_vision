import numpy as np
from PIL import Image
import keras.backend as K

# Function that change structure of dataset
# based on the channel ordering
# Takes as input an image dataset
def format_channel_ordering(dataset, ordering):
	new_dataset = []
	
	for img in dataset:
		# Need to find out if data is grayscale image; 
		if len(img.shape) < 3:
			if ordering == "channels_first":
				img = np.expand_dims(img, axis=0)
			if ordering == "channels_last":
				img = np.expand_dims(img, axis=2)
			new_dataset.append(img)
			continue

		if len(img.shape) > 2:
			if ordering == "channels_first":
				img = np.moveaxis(img, -1, 0)
			new_dataset.append(img)
			continue

	return new_dataset

if __name__ == "__main__":
	img = Image.open("penguin_parade.jpg")
	img = img.convert(mode='L')
	img = np.asarray(img)
	dataset = [img]

	ordering = K.image_data_format()
	print('Channel ordering: ', ordering)
	dataset = format_channel_ordering(dataset, ordering)
	print(dataset[0].shape)
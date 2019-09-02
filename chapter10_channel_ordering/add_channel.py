import numpy as np
from PIL import Image

img = Image.open('penguin_parade.jpg')
data = np.asarray(img)
print(data.shape)
# convert to grayscale
img = img.convert(mode='L')
data = np.asarray(img)
print(data.shape)

# add channels first
data_first = np.expand_dims(data, axis=0)
print(data_first.shape)

# add channels last
data_last = np.expand_dims(data, axis=2)
print(data_last.shape)
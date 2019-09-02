import numpy as np
from PIL import Image

img = Image.open('penguin_parade.jpg')
data = np.asarray(img)
print(data.shape)

# change from channels last to channels first
# move from last axis to first
data = np.moveaxis(data, -1, 0)
print(data.shape)

# change from channels first to channels last
data = np.moveaxis(data, 0, -1)
print(data.shape)
from keras.models import Sequential
from keras.layers import Input, Conv2D
import numpy as np

# Example of using bigger strides and the effect on the resulting feature map
# i.e. downsampling

# example 8x8 image
data = [
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 0, 0]]

data = np.asarray(data)
data = data.reshape((1, 8, 8, 1))

model = Sequential()
model.add(Conv2D(1, (3,3), strides=(2,2), input_shape=(8,8,1)))
model.summary()

# Vertical line detector
detector = [[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]]]

weights = [np.asarray(detector), np.asarray([0.0])]

model.set_weights(weights)

yhat = model.predict(data)

# Final output downsampled from 6x6 with stride of 1 compared to 3x3 with stride of 2
for r in range(yhat.shape[1]):
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

# Input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
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
model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8,8,1)))
model.add(MaxPooling2D())

model.summary()

# Define vertical line detector
detector = [[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]]]

weights = [np.asarray(detector), np.asarray([0.0])]

model.set_weights(weights)

yhat = model.predict(data)
for r in range(yhat.shape[1]):
  # print each column in the row
  print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
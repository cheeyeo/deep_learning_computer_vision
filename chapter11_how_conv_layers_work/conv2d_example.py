from keras.layers import Conv2D, Input
from keras.models import Model
import numpy as np

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

inputz = Input(shape=(8, 8, 1))
x = Conv2D(1, (3,3))(inputz)
model = Model(inputs=inputz, outputs=x)

# Define verical line detector
detector = [
[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]],
[[[0]],[[1]],[[0]]]
]

weights = [np.asarray(detector), np.asarray([0.0])]

model.set_weights(weights)

print('Model Weights: ', model.get_weights())

yhat = model.predict(data)
print(yhat)

for r in range(yhat.shape[1]):
  # print each column in the row
  print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
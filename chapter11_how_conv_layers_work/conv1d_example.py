from keras.layers import Conv1D, Input
from keras.models import Model
import numpy as np

data = np.asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape((1, 8, 1))

inputz = Input(shape=(8, 1))
x = Conv1D(filters=1, kernel_size=3)(inputz)
model = Model(inputs=inputz, outputs=x)

# Define verical line detector
detector = np.asarray([
[[0]], [[1]], [[0]]
])

print(detector.shape)

weights = [detector, np.asarray([0.0])]

model.set_weights(weights)

print('Model Weights: ', model.get_weights())

yhat = model.predict(data)
print(yhat)
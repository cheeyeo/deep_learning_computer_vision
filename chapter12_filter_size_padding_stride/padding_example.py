from keras.models import Sequential
from keras.layers import Conv2D

# Example of using same padding to preserve input shape for stacked CNNs
model = Sequential()
model.add(Conv2D(1, (3,3), padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(1, (3,3), padding='same'))
model.add(Conv2D(1, (3,3), padding='same'))
model.summary()

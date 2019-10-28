# Example of applying 1x1 filter to create a projection of the feature maps 
# Referred to as channel-wise pooling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
# Apply 1x1 projection here; note that the num of filters remain the same
model.add(Conv2D(512, (1,1), activation='relu'))
model.summary()
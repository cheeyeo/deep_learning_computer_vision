from keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D
from keras.models import Model
import keras.backend as K

def create_model(img):
	x = Conv2D(2, (3,3))(img)
	x = MaxPooling2D((2,2))(x)
	x = Flatten()(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(1, activation='softmax')(x)
	model = Model(inputs=img, outputs=x)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

print('[INFO] Channel Ordering: ', K.image_data_format())

if K.image_data_format() == 'channels_last':
	img = Input(shape=(28, 28, 1))
	model = create_model(img)
elif K.image_data_format() == 'channels_first':
	img = Input(shape=(1, 28, 28))
	model = create_model(img)

model.summary()
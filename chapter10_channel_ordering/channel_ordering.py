import keras.backend as K

print(K.image_data_format())

if K.image_data_format() == "channels_last":
	print("Channel last ordering...")
else:
	print("Channel first ordering...")

# Force channel_first ordering...
# Use 'th' for channel_first ordering; 'tf' for channel_last ordering
K.set_image_dim_ordering('th')
print(K.image_data_format())

# Force channel_last ordering
K.set_image_dim_ordering('tf')
print(K.image_data_format())
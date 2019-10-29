# Example of using VGG16 model as a classifier

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array


img = load_img("dog.jpg", target_size=(224, 224))

img = img_to_array(img)

img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

img = preprocess_input(img)

model = VGG16()

yhat = model.predict(img)

label = decode_predictions(yhat)
_, category, probs = label[0][0]

print("[INFO] Predicted: {}, Prob: {:.3f}%".format(category, probs*100))
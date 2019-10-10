# Train a SVM linear classifier on face embeddings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

data = np.load('faces-embeddings.npz')
trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('[INFO] Loaded embeddings: Train set: {}, Test set: {}'.format(trainX.shape, testX.shape))

# Normalize input vectors
# Embedding vectors compared to each other using distance metric
# Scale vectors until length or magnitude is 1 or unit length
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Encode the output labels categorically
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

# Create SVM model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)

print('Accuracy: Train={:.3f}, Test={:.3f}'.format(score_train*100, score_test*100))

# Test against random sample from test set
data2 = np.load('faces-dataset.npz')
testX_faces = data2['arr_2']

selection = random.choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_embed = testX[selection]
random_face_class = testY[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

samples = np.expand_dims(random_face_embed, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_prob = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('[INFO] Predicted: {}, {:.3f}'.format(predict_names[0], class_prob))
print('[INFO] Expected {}'.format(random_face_name[0]))

plt.imshow(random_face_pixels)
title = '{} {:.3f}'.format(predict_names[0], class_prob)
plt.title(title)
plt.savefig('test2.jpg')
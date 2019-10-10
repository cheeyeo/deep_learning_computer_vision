# Builds the training dataset
from utils import load_dataset
import numpy as np

trainX, trainY = load_dataset('train')
print(trainX.shape, trainY.shape)

testX, testY = load_dataset('val')
print(testX.shape, testY.shape)

np.savez_compressed('faces-dataset.npz', trainX, trainY, testX, testY)
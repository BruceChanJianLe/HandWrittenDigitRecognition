# Import the libraries
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model

import cv2
import matplotlib.pyplot as plt

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Understand the data structure
print('Before Data Processing')
print((X_train.shape, y_train.shape))
print((X_test.shape, y_test.shape))

# Pre-processing the data
# Note that mnist images only have a depth of 1, grayscale
num_class = 10
epochs = 7

# let the data be in the range of 0 and 1
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# Converting the output as 10 categories
y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

# After the data is being processed
print('After data process')
print((X_train.shape, y_train.shape))
print((X_test.shape, y_test.shape))

# Load trained model
cnn = load_model('trained_cnn_dropout_model.h5')

# Evaluate the model
score = cnn.evaluate(X_test, y_test)

# Print the result
print(score)

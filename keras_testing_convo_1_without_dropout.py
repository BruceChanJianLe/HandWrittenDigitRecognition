# Import the libraries
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

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
epochs = 20

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

# Creating the Convolution Neural Network
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same',
               activation='relu', name='Convolution_Layer_1'))
cnn.add(MaxPooling2D(pool_size=(2, 2), name='MaxPooling_Layer_1'))
cnn.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu',
               name='Convolution_Layer_2'))
cnn.add(MaxPooling2D(pool_size=(2, 2),name='MaxPooling_Layer_2'))
cnn.add(Flatten(name='Flattening_Layer'))
cnn.add(Dense(1024, activation='relu', name='Fully_Connected_Layer'))
cnn.add(Dense(10, activation='softmax', name='Output_Layer'))

# compile the CNN model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show the summary of the CNN model
cnn.summary()

# Training the CNN model
history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

# Show the accuracy
plt.plot(history_cnn.history['acc'])
plt.savefig('accuracy_cnn.png', bbox_inches='tight')
plt.plot(history_cnn.history['val_acc'])
plt.savefig('validation_accuracy_cnn.png', bbox_inches='tight')

# Saving the model for future use
cnn.save("trained_cnn_model.h5")

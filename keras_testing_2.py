from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


# Pre-processing the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Before reshaping the data')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Storing the height and width for reshaping use
img_height, img_width = X_train[0].shape # 28 * 28 = 784

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], img_height * img_width)
X_test = X_test.reshape(X_test.shape[0], img_height * img_width)

print('After reshaping the data')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Changing data to float format to convert the range to 0 to 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert to 0 to 1 by dividing 255 (largest possible)
X_train /= 255.0
X_test /= 255.0

# Organizing the output feature into 10 categories
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load the model
model = load_model("trained_model.h5")

score = model.evaluate(X_test, y_test)

print(score)

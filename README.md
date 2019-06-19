# Hand Written Digit Recognition

This project is to compare the performance of three different neural network.

- 1.Traditional Fully Connected Neural Network
- 2.Convolutional Neural Network
- 3.Convolutional Neural Network with Dropout layer of 20 percent

The MNIST database of handwritten digits are available in http://yann.lecun.com/exdb/mnist/

All the data are being pre-processed before feeding to three of the models

The data set are grayscale images, which is being normalize from 0 - 255 to 0 - 1

Also note that Keras NN takes in a 4 dimensional numpy array

It was originally a project done by Yann LeCun

## Performance in term of validation accuracy
  
### Traditional Fully Connected Neural Network
Testing Data Accuracy = 98.01%

### Convolutional Neural Network
Testing Data Accuracy = 99.18%

### Convolutional Neural Network with Dropout layer of 20 percent
Testing Data Accuracy = 99.32%

## Notes

### keras_testing_1.py
Traditional Fully Connected Neural Network file, training and saving process.
The result (weights) is being saved into trained_model.h5

### keras_tesing_2.py
This python3 script tests the trained model by loading the trained_model.h5 file.
And let it evaluate the testing data to obtain the accuracy of the model

### keras_testing_convo_1.py
Convolutional Neural Network file, with dropout layer, training and saving process.
The result (weights) is being saved into trained_cnn_dropout_model.h5

### keras_testing_convo_1_without_dropout.py
Convolutional Neural Network file, without dropout layer, training and saving process.
The result (weights) is being saved into trained_cnn_model.h5

### keras_testing_convo_2.py
This python3 script tests both trained_cnn_dropout_model.h5 and trained_cnn_model.h5.
It evaluates the testing date accuracy of both the models.

## Conclusion
We can tell from the testing data accuracy that the convolution neural network performs better in overall result.

It can be said that all three neural network performs almost the same but when the data sets gets larger and larger,
a few percent of difference does make a difference.

We can see that by adding 20 percent of dropout layer, the performance of the network increase.

# Others

## Resources
This is a course from lynda.com by Jonathan Fernandes "Neural Networks & Convolutional Neural Networks Essential Training"
## or create a new repository on the command line

echo "# HandWrittenDigitRecognition" >> README.md

git init

git add README.md

git commit -m "first commit"

git remote add HWDR https://github.com/BruceChanJianLe/HandWrittenDigitRecognition.git

git push -u HWDR master

## or push an existing repository from the command line

git remote add HWDR https://github.com/BruceChanJianLe/HandWrittenDigitRecognition.git

git push -u HWDR master
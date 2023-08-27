# -Deep-Learning-Algorithm
Deep Learning is a sub-field of Machine Learning that focuses on algorithms inspired by the structure and functions of the human brain calles as Artificial Neural Networks.
Artificial Neural Networks(ANN)- An ANN is a computational model that is inspired by the way biological neural networks in the human brain process information. It consists of interconnected artificial neurons, or nodes.
Neuron: The basic unit of computation in a neural network - it takes inputs, performs a computation on these inputs, and returns an output.
Weights and Biases: These are the learnable parameters of a neural network. They are updated during the learning process.
Activation Function: The activation function decides whether a neuron should be activated or not. It transforms the inputs of the neuron into its outputs. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.
Backpropagation: Backpropagation is a method used to train neural networks, where the error is calculated at the output and distributed back through the network layers.
Loss Function: A function that computes the difference between the network's prediction and the actual target. It guides the network in the right direction during training.
Optimizer: The optimizer uses the computed gradients of the loss function to update the network's parameters.
Epoch: One full pass through the entire training dataset.
Batch Size: The number of training examples utilized in one iteration.
Learning Rate: The size of the steps taken to reach the minimum of the loss function. If the learning rate is too large, the model might skip the minimum, while a small learning rate could slow down the training process.

# Deep Learning Model on MNIST Dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


network.compile(optimizer='rmsprop',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

predictions = network.predict(test_images)


# Convolutional Neural Networks
Convolutional Neural Networks (CNNs) are a type of neural network designed for processing structured grid data, such as images. CNNs have been successful in various tasks related to images, videos, and even audio. Here
Pooling Layer: Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information.
ReLU (Rectified Linear Unit) Layer: After each convolutional and fully-connected layer, a ReLU layer is often applied, which applies a non-linear function to the input, increasing the non-linear properties of the model and the overall network
Fully Connected Layer: Fully Connected layers in a neural networks are those layers where all the inputs from one layer are connected to every activation unit of the next layer. 
Dropout: Dropout is a regularization method where input and recurrent connections to a layer are probabilistically excluded from activation and weight updates while training a network, effectively reducing the number of parameters (albeit randomly). This technique improves generalization and reduces overfitting.



import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()


img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


    x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))




model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])





batch_size = 128
epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))              



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




# Transfer Learning

Transfer learning is a machine learning method where a model that was developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning because it allows us to build accurate models in a time-saving way

InceptionV3 model (pre-trained on the ImageNet dataset) to classify the CIFAR-10 dataset

import numpy as np
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

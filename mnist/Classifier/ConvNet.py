import keras

from keras.models import Sequential  # Build a sequential, linear-stack model
from keras.layers import Dense  # Fully connected layers
from keras.layers import Dropout  # Turns off random neurons for convergence
from keras.layers import Flatten  # Vectorize last hidden layers into output
from keras.layers import Conv2D  # Convolve over 2D images
from keras.layers import MaxPooling2D  # Pooling finds important features


class ConvNetBuilder():
    """
    Compile computational graph with Keras. Takes two arguments with several defaults:
        input_shape: Input tensor dimensions as tuple - An image could be (28, 28, 1)
        n_classes: Categorical classes - 10 classes for digit recognition task
        kernel_size: Default to (3,3) filter
        pool_size: Default to (2,2) field for max pooling layer

    """

    def __init__(self, input_shape, n_classes,
                 kernel_size=(3, 3),
                 pool_size=(2, 2)):
        super(ConvNetBuilder, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def compile_graph(self):
        model = Sequential()
        model.add(Conv2D(32,
                         kernel_size=self.kernel_size,
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64,
                         kernel_size=self.kernel_size,
                         activation='relu'))
        model.add(MaxPooling2D(self.pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

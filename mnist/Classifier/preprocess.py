from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical


class Preprocessor():
    """Preprocesses and stores MNIST data"""

    def __init__(self):
        super(Preprocessor, self).__init__()
        self.x_train, self.x_test = [], []
        self.y_train, self.y_test = [], []
        self.img_rows, self.img_cols = 28, 28
        self.num_classes = 10
        self.input_shape = ()

    def _load_mnist(self):
        train_set, test_set = mnist.load_data()
        self.x_train, self.y_train = train_set[0], train_set[1]
        self.x_test, self.y_test = test_set[0], test_set[1]

    def _reshape_mnist(self):
        # Check Keras 2D image format for reshaping tensors
        if K.image_data_format() == 'channels_first':
            # channel_first: (channels, rows, cols)
            self.x_train = self.x_train.reshape(
                self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(
                self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            # channel_last: (rows, cols, channels)
            self.x_train = self.x_train.reshape(
                self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(
                self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

    def format_mnist(self):
        # Get MNIST and input shape
        self._load_mnist()
        self._reshape_mnist()

        # Cast training data to float32
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # Compute grayscale pixel values; 0 to 255
        self.x_train /= 255
        self.x_test /= 255

        # Factorize the class labels
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

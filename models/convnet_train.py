import keras
from keras import backend  # Will use a TF backend
from keras.datasets import mnist  # MNIST dataset of 28x28 pixel b+w images
from keras.models import Sequential  # Build a sequential, linear-stack model
from keras.layers import Dense  # Fully connected layers
from keras.layers import Dropout  # Turns off random neurons for convergence
from keras.layers import Flatten  # Vectorize last hidden layers into output
from keras.layers import Conv2D  # Convolve over 2D images
from keras.layers import MaxPooling2D  # Pooling finds important features


def build_model(input_shape, num_classes):
    model = Sequential()  # Initialize Sequential class
    model.add(Conv2D(32,  # Integer of output filters in the convolution
                     kernel_size=(3, 3),  # Dims of 2D convolution window
                     activation='relu',  # Rectified Linear Unit activiation
                     input_shape=input_shape))  # 4D Tensor in and out
    model.add(Conv2D(64,  # Input is the output of the previous conv layer
                     kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Max element in feature space
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def save_model(model):
    # Save jSON file first
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize model weights to HDF5
    model.save_weights("models/model.h5")
    print("Model and Weights Saved.")


def main():
    # Hyperparameters of the Architecture
    batch_size = 128  # Mini-batch gradient descent size
    num_classes = 10  # 10 digits to classify, 0 to 9
    epochs = 12  # Start training at 12 'epochs'

    # Load MNIST dataset which is split into train/test and examples/labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28  # Each image resolution is 28x28 pixels

    # Check Keras 2D image format for reshaping tensors
    if backend.image_data_format() == 'channels_first':
        # channel_first: (channels, rows, cols)
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        # channel_last: (rows, cols, channels)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Cast training data to float32
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Compute grayscale pixel values; 0 to 255
    x_train /= 255
    x_test /= 255

    # Training set is a (60000, 28, 28, 1) 4D tensor
    print('\nx_train shape: ', x_train.shape)
    print(x_train.shape[0], ' training samples')
    print(x_test.shape[0], ' test samples')

    # Factorize the class labels
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Compile and train model
    model = build_model(input_shape, num_classes)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Error: ', score[0])
    print('Accuracy: ', score[1])

    # Save model weights to disk
    save_model(model)

if __name__ == '__main__':
    main()

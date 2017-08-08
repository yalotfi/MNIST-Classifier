import numpy as np

from keras.models import model_from_json
from PIL import Image


def load(json_path, weight_path):
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weight_path)
    print("Loaded model architecture and weights from disk")
    return model


def save_model(model, architecture_path, weight_path):
    # Save architecture in jSON file first
    model_json = model.to_json()
    with open(architecture_path, 'w') as json_file:
        json_file.write(model_json)

    # Serialize model weights to HDF5
    model.save_weights(weight_path)
    print("Model architecture and weights saved to disk")


def process_img(img_path, res=(28, 28)):
    # Read, convert to grayscale, and resize the image file
    im = Image.open(img_path)
    im = im.convert(mode='L')
    print("Converted img Size: {}".format(im.size))
    im = im.resize(res, resample=Image.NEAREST)
    print("Resolution: {} \nColor Mode: {}".format(im.size, im.mode))

    # Reshape image into 4D tensor
    tensor = np.array(im.getdata()).reshape(1, res[0], res[1], 1)
    print(tensor.shape)
    im.close()
    return tensor


def main():
    tensor = process_img('test/mnist2.png')
    model = load('app/model/architecture.json',
                 'app/model/weights.h5')

    probabilities = model.predict(tensor)
    print(probabilities)
    prediction = model.predict_classes(tensor)
    print(prediction)


if __name__ == '__main__':
    main()

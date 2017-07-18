import numpy as np
import base64

from flask import Flask, jsonify, request, render_template
from keras.models import model_from_json
from io import BytesIO
from PIL import Image


def load(json_path, weight_path):
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weight_path)
    print("Loaded model architecture and weights from disk")
    return model


def process_img(data, res=(28, 28)):
    # Read, convert to grayscale, and resize the image file
    im = Image.open(
        BytesIO(base64.b64decode(
            data)
        )
    )
    im = im.convert(mode='L')
    im = im.resize(res, resample=Image.NEAREST)
    print("Resolution: {0} \nColor Mode: {1}").format(im.size, im.mode)

    # Reshape image into 4D tensor
    tensor = np.array(im.getdata()).reshape(1, res[0], res[1], 1)
    print(tensor.shape)
    im.close()
    return tensor


app = Flask(__name__)
app.debug = True


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)


@app.route('/upload', methods=['POST'])
def upload():
    data = request.form['img']
    model_input = process_img(data)
    model = load('../model/architecture.json', '../model/weights.h5')
    label = model.predict_classes(model_input)
    return jsonify(status='got image', number=label.tolist()[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)

import numpy as np
import base64
from PIL import Image
from io import BytesIO

from flask import render_template
from flask import request
from flask import jsonify

from Classifier.ModelIO import load_trained_model
from app import app

# Load model for inference
architecture_path = 'model/architecture.json'
weight_path = 'model/weights.h5'
model = load_trained_model(architecture_path, weight_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Get json data from user
    data = request.get_json('img')
    img_base64 = data['img']

    # Process image from byte64 encoding -> image -> tensor
    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    img = img.convert(mode='L')
    img = img.resize((28, 28), resample=Image.NEAREST)
    tensor = np.array(img.getdata()).reshape(1, 28, 28, 1)
    img.close()

    # Inference time: POST probability vector and predicted class
    probs = model.predict(tensor)
    preds = model.predict_classes(tensor)
    return jsonify(status='got img',
                   probabilities=probs,
                   predicted_class=preds)

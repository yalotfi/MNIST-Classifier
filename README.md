# Web App:
### Running the App

1. Run a virtual enviornment and install dependencies

#### Ubuntu/OSX
```
$ virtualenv mnist-env
$ source mnist-env/bin/activate
$ pip install -r requirements.txt
```
#### Windows
```
> virtualenv mnist-env
> mnist-env/Scripts/activate
> pip install -r requirements.txt
```

2. Run the flask app

Easiest way is to simply execute `python run.py` in the root directory. Syntax depends on your Python interpreter, but run this script as you would any Python program.

### What's the Stack?

1. TensorFlow - Backend Deep Learning Library for Keras
2. Keras - High level API to quickly build and train the ConvNet
3. Flask - Serving the machine learning model
4. Jinja - HTML templating
5. p5.js - Graphical and interactive JS library based on Processing

# Model:
### Training:
`train.py` in `app/` will train the model. On a GTX970 GPU, each epoch takes 8s. On an Intel i5 CPU, runtime shoots to about 30s per epoch.

### ConvNet Architecture:

| Layer			| Memory		| Weights			|
| ---------------------	| ---------------------	| -----------------------------	|
| INPUT: [28x28x1]	| memory: 28x28x1=784 	| weights: 0			|
| CONV2-32: [28x28x32]	| memory: 28x28x32=25088| weights:(3x3x1)x32=288	|
| CONV2-64: [28x28x64]	| memory: 28x28x64=50176| weights:(3x3x1)x64=576	|
| POOL2: [14x14x64]	| memory: 14x14x64=12544| weights: 0			|
| FC: [1x1x128]		| memory: 128		| weights: 14x14x64x128=1605632 |
| FC: [1x1x10]		| memory: 10		| weights: 128x10=1280		|

### HyperParameters
```
Convolutions: 3x3
MaxPool-Field: 2x2
Stride: 1
Padding: none
Dropout: [0.25, 0.5]
Activation: [ReLu, ReLu, Softmax]
Loss-Function: Cross-Entropy
Optimization: AdaDelta
```


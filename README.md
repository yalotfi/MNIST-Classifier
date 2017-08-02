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

```
export FLASK_APP=mnist
pip install -e .
flask run
```

### What's the Stack?

1. Flask - Serving the machine learning model
2. Jinja - HTML templating
3. p5.js - Graphical and interactive JS library based on Processing

# Model:
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


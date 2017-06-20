## MNIST Classification

Train a Keras-based Convolutional Neural Net to perform hand-written digit classification. This model is using a TensorFlow backend.

## Architecture:

INPUT: [28x28x1]	memory: 28*28*1=784	weights: 0
CONV2-32: [28x28x32]	memory: 28*28*32=25088	weights:(3*3*1)*32=288
CONV2-64: [28x28x64]	memory: 28*28*64=50176	weights:(3*3*1)*64=576
POOL2: [14x14x64]	memory: 14*14*64=12544	weights: 0
FC: [1x1x128]	memory: 128	weights: 14*14*64*128=1605632
FC: [1x1x10]	memory: 10	weights: 128*10=1280

## HyperParameters

Convolutions: 3x3
MaxPool-Field: 2x2
Stride: 1
Padding: none
Dropout: [0.25, 0.5]
Activation: [ReLu, ReLu, Softmax]
Loss-Function: Cross-Entropy
Optimization: AdaDelta



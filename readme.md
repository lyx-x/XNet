# XNet: simple CuDNN wrapper

XNet is still under development, some parts may not be completed yet.

## About

This project is part of my undergraduate research project "Object classification with Convolutional Neural Network". It is inspired by Mr. Hinton's paper on 2012 which is included in the repository. Unlike [Caffe](http://caffe.berkeleyvision.org/), [Theano](http://deeplearning.net/software/theano/) or [Torch](http://torch.ch/), XNet is just a basic wrapper with few customizations and not very stable, but it is simple and easily understandable. You may want to use stable framework mentionned above for any production purposes.

Traditionnal training example like MNIST and CIFAR will be added in the future.

All suggestions are welcomed and hope that this piece of code is useful to you.

## Included features

* Layers
  * Convolutional layer
  * Max-pooling layer
  * Fully connected layer (with ReLU or Softmax)
  * Activation layer
  * Input / Output layer
* Neural Network
  * Adjustable layer dimension 
  * Adjustable learning rate for each layer
  * Decreasing learning rate
  * Momentum and weight decay
* Examples
  * MNIST
    * Test error rate: 0.5% - 1%
    * Real-time recognition on camera with OpenCV
  * CIFAR-10
    * Test error rate: around 25% (with data augmentation)

## How to use

To run this code, you will need:

1. NVIDIA graphic card 
2. Cuda Runtime 7.0
3. CuDNN library 3
4. C++ 11 compiler
5. OpenCV 3.0 (to use your camera)
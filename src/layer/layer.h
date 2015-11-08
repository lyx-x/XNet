/*
 * layer.h
 *
 * Layer class, this class is abstract, it provides basic layer members like
 * data and some methods. All layers should extend this class.
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../utils/global.h"
#include "../utils/set_value.h"
#include "../utils/print.h"
#include "../utils/utils.h"

namespace layer {

class Layer {
public:
	Layer(float alpha = 0, float momentum = 0.9f, float weight_decay = 0);
	virtual ~Layer();

	// three virtual method that all layers should have
	virtual void forward(bool train = true) = 0;
	virtual void backward() = 0;
	virtual void update() = 0;

	void adjust_learning(float scale);

	Layer* prev; // previous layer
	Layer* next; // next layer
	cudnnTensorDescriptor_t t_data; // output dimension
	float* data; // output
	int data_size; // output size
	float* diff; // difference for the previous layer
	float* param; // parameters
	int param_size; // parameters count
	float* param_bias; // bias parameters for some layers
	int param_bias_size; // bias parameters count
	float* gradient; // gradient of parameters
	float* gradient_bias; // gradient of bias parameters
	int batch; // batch size
	float alpha; // learning rate
	float momentum; // momentum of gradient
	float weight_decay; // weight decay rate
};

} /* namespace layer */
#endif /* LAYER_H_ */

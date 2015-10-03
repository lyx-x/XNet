/*
 * layer.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef LAYER_CUH_
#define LAYER_CUH_

#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../utils/global.h"
#include "../utils/set_value.cuh"
#include "../utils/print.cuh"
#include "../utils/utils.h"

namespace layer {

class Layer {
public:
	Layer(float alpha = 0);
	virtual ~Layer();

	// three virtual method that all layers should have
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void update() = 0;

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
	float alpha;
	bool isProduction;
};

} /* namespace layer */
#endif /* LAYER_CUH_ */

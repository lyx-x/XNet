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
	Layer();
	virtual ~Layer();

	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void update(float alpha) = 0;

	Layer* prev;
	Layer* next;
	cudnnTensorDescriptor_t t_data;
	float* data;
	int data_size;
	float* diff;
	float* param;
	int param_size;
	float* param_bias;
	int param_bias_size;
	float* gradient;
	float* gradient_bias;
	int batch;
};

} /* namespace layer */
#endif /* LAYER_CUH_ */

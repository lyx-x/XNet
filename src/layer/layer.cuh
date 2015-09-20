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
	virtual void update() = 0;

	Layer* prev;
	Layer* next;
	cudnnTensorDescriptor_t t_data;
	float* data;
	float* diff;
	float* param;
	float* param_bias;
	float* gradient;
	int batch;
};

} /* namespace layer */
#endif /* LAYER_CUH_ */

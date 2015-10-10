/*
 * activation.cpp
 *
 *  Created on: Oct 5, 2015
 *      Author: lyx
 */

#include "activation.h"

using namespace global;

namespace layer {

Activation::Activation(Layer* _prev, cudnnActivationMode_t _mode) : Layer(0) {
	prev = _prev;
	prev->next = this;

	mode = _mode;

	int _n, _c, _h, _w, _tmp;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));
	batch = _n;
	data_size = _n * _c * _h * _w;
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			_n, _c, _h, _w));
	callCuda(cudaMalloc(&data, sizeof(float) * data_size));
	callCuda(cudaMalloc(&diff, sizeof(float) * data_size));

	param_size = 0;
	param_bias_size = 0;
}

Activation::~Activation() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
}

void Activation::forward(bool train) {
	float a = 1;
	float b = 0;
	callCudnn(cudnnActivationForward(cudnnHandle, mode, &a, prev->t_data,
			prev->data, &b, t_data, data));
}

void Activation::backward() {
	float a = 1;
	float b = 0;
	callCudnn(cudnnActivationBackward(cudnnHandle, mode, &a, t_data, data, t_data,
			next->diff, prev->t_data, prev->data, &b, t_data, diff));
}

void Activation::update() {
	// nothing
}

} /* namespace layer */

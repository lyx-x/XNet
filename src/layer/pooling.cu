/*
 * pooling.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "pooling.cuh"

namespace layer {

Pooling::Pooling(Layer& _prev, int size, int stride) {
	prev = _prev;
	_prev.next = this;

	callCudnn(cudnnSetPooling2dDescriptor(descriptor, CUDNN_POOLING_MAX,
			size, size,	0, 0, stride, stride));

	int _n, _c, _h, _w, _tmp;
	batch = _n;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev.t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));
	checkCUDNN(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			_n, _c, _h / stride, _w / stride));
	int data_size = _n * _c * (_h / stride) * (_w / stride);
	callCuda(cudaMalloc(data, sizeof(float) * data_size));
	callCuda(cudaMalloc(diff, sizeof(float) * data_size));
}

Pooling::~Pooling() {
	callCudnn(cudnnDestroyPoolingDescriptor(descriptor));
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
}

void Pooling::forward() {
	float a = 0;
	float b = 0;
	callCudnn(cudnnPoolingForward(cudnnHandle, descriptor, &a, prev.t_data,
			prev.data, &b, t_data, data));
}

void Pooling::backward() {

}

void Pooling::update() {

}

}
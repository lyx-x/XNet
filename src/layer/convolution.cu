/*
 * convolution.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "convolution.cuh"

using namespace global;

namespace layer {

Convolution::Convolution(Layer& _prev, int n ,int c, int h, int w, int kernel) :
		Layer() {
	prev = _prev;
	_prev.next = this;

	callCudnn(cudnnSetConvolution2dDescriptor(descriptor, 0, 0,	1, 1, 1, 1,
			CUDNN_CROSS_CORRELATION));

	int _n, _c, _h, _w, _tmp;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev.t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));
	callCudnn(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT,
			c, _c, kernel, kernel));
	int param_size =  _c * c * kernel * kernel;
	callCuda(cudaMalloc(param, sizeof(float) * param_size));
	callCuda(cudaMalloc(gradient, sizeof(float) * param_size));
	utils::setGpuNormalValue(param, param_size);

	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW,	CUDNN_DATA_FLOAT,
			n, c, h, w));
	callCuda(cudaMalloc(data, sizeof(float) * n * c * h * w));
	callCuda(cudaMalloc(diff, sizeof(float) * n * c * h * w));

	callCudnn(cudnnCreateTensorDescriptor(&t_bias));
	callCudnn(cudnnSetTensor4dDescriptor(t_bias, CUDNN_TENSOR_NCHW,	CUDNN_DATA_FLOAT,
			1, c, 1, 1));
	callCuda(cudaMalloc(param_bias, sizeof(float) * c));
	utils::setGpuNormalValue(param_bias, c);

	callCudnn(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, prev.t_data, filter,
			descriptor, t_data,	CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
	callCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			prev.t_data, filter, descriptor, t_data, algo, &workspace_size));
}

Convolution::~Convolution() {
	callCudnn(cudnnDestroyFilterDescriptor(filter));
	callCudnn(cudnnDestroyConvolutionDescriptor(descriptor));
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCudnn(cudnnDestroyTensorDescriptor(t_bias));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
	callCuda(cudaFree(param));
	callCuda(cudaFree(param_bias));
	callCuda(cudaFree(gradient));
}

void Convolution::forward() {

}

void Convolution::backward() {

}

void Convolution::update() {

}

}

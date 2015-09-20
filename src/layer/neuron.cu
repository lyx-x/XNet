/*
 * neuron.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "neuron.cuh"

using namespace global;

namespace layer {

Neuron::Neuron(Layer* _prev, int _output_size) {
	prev = _prev;
	prev->next = this;

	int _n, _c, _h, _w, _tmp;
	cudnnDataType_t _t;
	callCudnn(cudnnGetTensor4dDescriptor(prev->t_data, &_t, &_n, &_c, &_h, &_w, &_tmp,
			&_tmp, &_tmp, &_tmp));
	batch = _n;
	input_size = _c;
	output_size = _output_size;
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			_n, output_size, 1, 1));
	callCuda(cudaMalloc(&data, sizeof(float) * _n * output_size));
	callCuda(cudaMalloc(&diff, sizeof(float) * _n * output_size));

	callCuda(cudaMalloc(&param, sizeof(float) * _c * output_size));
	callCuda(cudaMalloc(&param_bias, sizeof(float) * output_size));
	callCuda(cudaMalloc(&gradient, sizeof(float) * _c * output_size));

	callCuda(cudaMalloc(&one, sizeof(float) * batch));

}

Neuron::~Neuron() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(diff));
	callCuda(cudaFree(param));
	callCuda(cudaFree(param_bias));
	callCuda(cudaFree(gradient));
	callCuda(cudaFree(one));
}

void Neuron::forward() {
	float a = 1;
	float b = 0;
	callCuda(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, output_size, batch,
			input_size,	&a, param, input_size, prev->data, input_size, &b, data,
			output_size));
	callCuda(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, batch,
			1, &a, param_bias, output_size,	one, 1,	&a,	data, output_size));
}

void Neuron::backward() {

}

void Neuron::update() {

}

}

/*
 * output.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "output.cuh"

using namespace global;

namespace layer {

Output::Output(Layer* _prev, float* _label, int n) : Layer() {
	prev = _prev;
	prev->next = this;

	batch = n;
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			n, 1, 1, 1));
	callCuda(cudaMalloc(&data, sizeof(float) * n));
	callCuda(cudaMalloc(&label, sizeof(float) * n));
	callCuda(cudaMemcpy(label, _label, n, cudaMemcpyHostToDevice));
}

Output::~Output() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(label));
}

void Output::forward() {
	// nothing
}

void Output::backward() {

}

void Output::update() {
	// nothing
}

}

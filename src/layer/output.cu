/*
 * output.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "output.cuh"

using namespace global;

namespace layer {

__global__ void softmaxLoss(const float *label, int labels, int batch, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch)
		return;

	const int label_value = static_cast<int>(label[idx]);

	diff[idx * labels + label_value] -= 1.0f;
}

Output::Output(Layer* _prev, float* _label, int _labels, int _batch) : Layer() {
	prev = _prev;
	prev->next = this;

	batch = _batch;
	labels = _labels;
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			batch, 1, 1, 1));
	data_size = batch;
	callCuda(cudaMalloc(&data, sizeof(float) * data_size));
	callCuda(cudaMalloc(&label, sizeof(float) * data_size));
	callCuda(cudaMemcpy(label, _label, batch, cudaMemcpyHostToDevice));

	callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));
}

Output::~Output() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	callCuda(cudaFree(label));
	callCuda(cudaFree(diff));
}

void Output::forward() {
	// nothing
}

void Output::backward() {
	softmaxLoss<<< (batch + 127) / 128, 128>>> (label, labels, batch, diff);
}

void Output::update(float alpha) {
	// nothing
}

}

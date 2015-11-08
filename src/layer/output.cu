/*
 * output.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "output.h"

using namespace global;

namespace layer {

__global__ void softmaxLoss(const float *label, int label_dim, int batch, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch)
		return;

	const int label_value = static_cast<int>(label[idx]);

	diff[idx * label_dim + label_value] -= 1.0f;
}

__global__ void predict(const float *softmax, int label_dim, int batch, float *data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch)
		return;

	int label_value = 0;
	float max = -1;
	for (int i = 0; i < label_dim; i++) {
		if (softmax[idx * label_dim + i] > max) {
			max = softmax[idx * label_dim + i];
			label_value = i;
		}
	}

	data[idx] = (float)label_value;
}

Output::Output(Layer* _prev, float* _label, int _label_dim, int _batch) : Layer() {
	prev = _prev;
	prev->next = this;

	batch = _batch;
	label_dim = _label_dim;
	callCudnn(cudnnCreateTensorDescriptor(&t_data));
	callCudnn(cudnnSetTensor4dDescriptor(t_data, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			batch, 1, 1, 1));
	data_size = batch;
	callCuda(cudaMalloc(&data, sizeof(float) * data_size));
	label = _label;

	callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));

	param_size = 0;
	param_bias_size = 0;
}

Output::~Output() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	label = NULL;
	callCuda(cudaFree(diff));
}

void Output::forward(bool train) {
	predict<<< (batch + 127) / 128, 128>>> (prev->data, label_dim, batch, data);
}

void Output::backward() {
	callCuda(cudaMemcpy(diff, prev->data, sizeof(float) * prev->data_size,
			cudaMemcpyDeviceToDevice));
	softmaxLoss<<< (batch + 127) / 128, 128>>> (label, label_dim, batch, diff);
}

void Output::update() {
	// nothing
}

}

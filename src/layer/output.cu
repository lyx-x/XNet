/*
 * output.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "output.cuh"

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
	//callCuda(cudaMalloc(&label, sizeof(float) * data_size));
	//callCuda(cudaMemcpy(label, _label, batch, cudaMemcpyHostToDevice));
	label = _label;

	callCuda(cudaMalloc(&diff, sizeof(float) * prev->data_size));
}

Output::~Output() {
	callCudnn(cudnnDestroyTensorDescriptor(t_data));
	callCuda(cudaFree(data));
	//callCuda(cudaFree(label));
	label = NULL;
	callCuda(cudaFree(diff));
}

void Output::forward() {
	// nothing
}

void Output::backward() {
	//std::cout << "hehre\n";
	utils::printGpuMatrix(label, batch, 1, batch, 2);
	callCuda(cudaMemcpy(diff, prev->data, sizeof(float) * prev->data_size,
			cudaMemcpyDeviceToDevice));
	softmaxLoss<<< (batch + 127) / 128, 128>>> (label, label_dim, batch, diff);
	utils::printGpuMatrix(diff, batch * 10, 10, batch, 2);
}

void Output::update(float alpha) {
	// nothing
}

}

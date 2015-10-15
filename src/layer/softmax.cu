/*
 * softmax.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "softmax.cuh"

using namespace global;

namespace layer {

Softmax::Softmax(Layer* _prev, int _output_size, float dropout_rate, float alpha) :
		Neuron(_prev, _output_size, dropout_rate, alpha) {}

Softmax::~Softmax() {}

void Softmax::forward_activation() {
	float a = 1;
	float b = 0;
	callCudnn(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL, &a, t_data, tmp_data, &b, t_data, data));
}

void Softmax::backward_activation() {
	float a = 1.0 / batch;
	float b = 0;
	callCudnn(cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,	&a, t_data, data, t_data, next->diff,
			&b, t_data, tmp_diff));
	//utils::setGpuValue(tmp_diff, data_size, 0);
	//callCuda(cublasSaxpy(cublasHandle, data_size, &a, next->diff, 1, tmp_diff, 1));
	//utils::printGpuMatrix(next->diff, 20, 10, 2, 4);
}

}

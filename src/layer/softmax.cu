/*
 * softmax.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "softmax.cuh"

using namespace global;

namespace layer {

Softmax::Softmax(Layer* _prev, int _output_size, float dropout_rate, float alpha,
		float sigma) : Neuron(_prev, _output_size, dropout_rate, alpha, sigma) {}

Softmax::~Softmax() {}

void Softmax::forward_activation() {
	float a = 1;
	float b = 0;
	callCudnn(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_CHANNEL, &a, t_data, tmp_data, &b, t_data, data));
}

void Softmax::backward_activation() {
	float a = 1.0 / batch;
	float b = 0;
	callCudnn(cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_FAST,
			CUDNN_SOFTMAX_MODE_CHANNEL,	&a, t_data, data, t_data, next->diff,
			&b, t_data, tmp_diff));
}

}

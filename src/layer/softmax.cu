/*
 * softmax.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "softmax.cuh"

using namespace global;

namespace layer {

Softmax::Softmax(Layer* _prev, int _output_size) : Neuron(_prev, _output_size) {}

Softmax::~Softmax() {}

void Softmax::forward_activation() {
	float a = 1;
	float b = 0;
	callCudnn(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL, &a, t_data, tmp_data, &b, t_data, data));
}

void Softmax::backward_activation() {

}

}
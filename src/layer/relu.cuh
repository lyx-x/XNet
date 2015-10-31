/*
 * relu.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef RELU_CUH_
#define RELU_CUH_

#include "neuron.cuh"

namespace layer {

class ReLU: public Neuron {
public:
	ReLU(Layer* _prev, int _output_size, float dropout_rate, float alpha,
			float sigma = 0.01f);
	virtual ~ReLU();
	void forward_activation();
	void backward_activation();
};

} /* namespace layer */
#endif /* RELU_CUH_ */

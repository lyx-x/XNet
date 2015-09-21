/*
 * neuron.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef NEURON_CUH_
#define NEURON_CUH_

#include "layer.cuh"

namespace layer {

class Neuron : public Layer {
protected:
	float* tmp_data;
	float* tmp_diff;
	int input_size;
	int output_size;
	float* one;
public:
	Neuron(Layer* _prev, int _output_size);
	virtual ~Neuron();
	void forward();
	virtual void forward_activation() = 0;
	void backward();
	virtual void backward_activation() = 0;
	void update(float alpha);
};

} /* namespace layer */
#endif /* NEURON_CUH_ */

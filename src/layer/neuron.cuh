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

// fully connected layer
class Neuron : public Layer {
protected:
	float* tmp_data; // z
	float* tmp_diff; // difference of z
	int input_size; // output size of previous layer
	int output_size; // output size
	float* one; // full one vector for bias
public:
	Neuron(Layer* _prev, int _output_size); // data is output_size * batch
	virtual ~Neuron();
	void forward();
	virtual void forward_activation() = 0; // support different activation function
	void backward();
	virtual void backward_activation() = 0;
	void update(float alpha);
};

} /* namespace layer */
#endif /* NEURON_CUH_ */

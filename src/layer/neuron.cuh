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
private:
	float dropout_rate;
protected:
	float* tmp_data; // z
	float* tmp_diff; // difference of z
	int input_size; // output size of previous layer
	int output_size; // output size
	float* one; // full one vector for bias
	void dropout(bool train);
public:
	// data is output_size * batch
	Neuron(Layer* _prev, int _output_size, float dropout_rate, float alpha, float sigma = 0.01f);
	virtual ~Neuron();
	void forward(bool train = true);
	virtual void forward_activation() = 0; // support different activation function
	void backward();
	virtual void backward_activation() = 0;
	void update();
};

} /* namespace layer */
#endif /* NEURON_CUH_ */

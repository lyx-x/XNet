/*
 * neuron.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef NEURON_CUH_
#define NEURON_CUH_

namespace layer {

class Neuron : public Layer{
private:
	int input_size;
	int output_size;
	float* one;
public:
	Neuron(Layer* _prev, int _output_size);
	virtual ~Neuron();
	void forward();
	void backward();
	void update();
};

} /* namespace layer */
#endif /* NEURON_CUH_ */

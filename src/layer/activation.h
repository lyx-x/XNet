/*
 * activation.h
 *
 *  Created on: Oct 5, 2015
 *      Author: lyx
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "layer.cuh"

namespace layer {

class Activation: public Layer {
private:
	cudnnActivationMode_t mode;
public:
	Activation(Layer* prev, cudnnActivationMode_t mode);
	virtual ~Activation();

	void forward();
	void backward();
	void update();
};

} /* namespace layer */
#endif /* ACTIVATION_H_ */

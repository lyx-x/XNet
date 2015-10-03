/*
 * pooling.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef POOLING_CUH_
#define POOLING_CUH_

#include "layer.cuh"

namespace layer {

class Pooling : public Layer {
private:
	cudnnPoolingDescriptor_t descriptor;
public:
	Pooling(Layer* _prev, int size, int stride);
	virtual ~Pooling();
	void forward();
	void backward();
	void update();
};

} /* namespace layer */
#endif /* POOLING_CUH_ */

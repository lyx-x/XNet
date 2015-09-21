/*
 * input.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef INPUT_CUH_
#define INPUT_CUH_

#include "layer.cuh"

namespace layer {

class Input: public Layer {
public:
	Input(int n, int c, int h, int w);
	virtual ~Input();
	void forward();
	void backward();
	void update(float alpha);
};

} /* namespace layer */
#endif /* INPUT_CUH_ */

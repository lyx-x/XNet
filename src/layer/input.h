/*
 * input.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef INPUT_CUH_
#define INPUT_CUH_

#include "layer.h"

namespace layer {

class Input: public Layer {
public:
	Input(int n, int c, int h, int w, float* _data);
	virtual ~Input();
	void forward(bool train = true);
	void backward();
	void update();
};

} /* namespace layer */
#endif /* INPUT_CUH_ */

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
	Input();
	virtual ~Input();
};

} /* namespace layer */
#endif /* INPUT_CUH_ */

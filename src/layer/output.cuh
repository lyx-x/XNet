/*
 * output.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef OUTPUT_CUH_
#define OUTPUT_CUH_

#include "layer.cuh"

namespace layer {

class Output: public Layer {
public:
	Output();
	virtual ~Output();
};

} /* namespace layer */
#endif /* OUTPUT_CUH_ */

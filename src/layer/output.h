/*
 * output.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef OUTPUT_CUH_
#define OUTPUT_CUH_

#include "layer.h"

namespace layer {

class Output: public Layer {
public:
	float* label; // real label
	int label_dim; // dimension of label (ex. 10 for digit recognition)
public:
	Output(Layer* _prev, float* _label, int _label_dim, int _batch);
	virtual ~Output();
	void forward(bool train = true);
	void backward();
	void update();
};

} /* namespace layer */
#endif /* OUTPUT_CUH_ */

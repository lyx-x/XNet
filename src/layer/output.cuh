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
private:
	float* label;
	int labels;
public:
	Output(Layer* _prev, float* _label, int _labels, int _batch);
	virtual ~Output();
	void forward();
	void backward();
	void update(float alpha);
};

} /* namespace layer */
#endif /* OUTPUT_CUH_ */

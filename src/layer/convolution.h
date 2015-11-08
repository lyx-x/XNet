/*
 * convolution.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include "layer.h"

namespace layer {

class Convolution : public Layer {
private:
	cudnnFilterDescriptor_t filter;
	cudnnConvolutionDescriptor_t descriptor;
	cudnnTensorDescriptor_t t_bias;
	cudnnConvolutionFwdAlgo_t algo;
	size_t workspace_size; // extra size for computing
	void* workspace; // pointer to the extra size
	float* tmp_data;
	float* tmp_diff;
public:
	Convolution(Layer* _prev, int n ,int c, int kernel, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);
	virtual ~Convolution();
	void forward(bool train = true);
	void backward();
	void update();
};

} /* namespace layer */
#endif /* CONVOLUTION_CUH_ */

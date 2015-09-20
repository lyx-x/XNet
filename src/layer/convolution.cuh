/*
 * convolution.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include "layer.cuh"

namespace layer {

class Convolution : public Layer {
private:
	cudnnFilterDescriptor_t filter;
	cudnnConvolutionDescriptor_t descriptor;
	cudnnTensorDescriptor_t t_bias;
	cudnnConvolutionFwdAlgo_t algo;
	size_t workspace_size;
	void* workspace;
public:
	Convolution(Layer* _prev, int n ,int c, int h, int w, int kernel);
	virtual ~Convolution();
	void forward();
	void backward();
	void update();
};

} /* namespace layer */
#endif /* CONVOLUTION_CUH_ */

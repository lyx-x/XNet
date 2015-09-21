/*
 * network.h
 *
 *  Created on: Sep 21, 2015
 *      Author: lyx
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <iostream>
#include <vector>

#include "../layer/layer.cuh"
#include "../layer/input.cuh"
#include "../layer/output.cuh"
#include "../layer/convolution.cuh"
#include "../layer/pooling.cuh"
#include "../layer/relu.cuh"
#include "../layer/softmax.cuh"

using namespace layer;

namespace model {

class Network {
private:
	std::vector<Layer*> layers;
	float* data;
	float* label;
	int size, batch;
public:
	Network(float* data, float* label, int size, int batch);
	virtual ~Network();
	void Train(int iteration, float alpha);
	void PushInput(int n, int c, int h, int w, float* data);
	void PushOutput(float* label, int num_label, int batch);
	void PushConvolution(int n ,int c, int h, int w, int kernel);
	void PushPooling(int size, int stride);
	void PushReLU(int output_size);
	void PushSoftmax(int output_size);
	void Pop();
};

} /* namespace model */
#endif /* NETWORK_H_ */

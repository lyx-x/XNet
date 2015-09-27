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
#include <typeinfo>

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
	float* h_data;
	int data_dim;
	float* label;
	float* h_label;
	int label_dim;
	int size, batch;
public:
	Network(float* data, int data_dim, float* label, int label_dim,
			int count, int batch);
	virtual ~Network();
	void Train(int iteration, float alpha);
	void PushInput(int c, int h, int w);
	void PushOutput(int label_dim);
	void PushConvolution(int c, int kernel);
	void PushPooling(int size, int stride);
	void PushReLU(int output_size);
	void PushSoftmax(int output_size);
	void Pop();

	void PrintGeneral();
	void PrintData(int offset, int r, int c, int precision);
};

} /* namespace model */
#endif /* NETWORK_H_ */

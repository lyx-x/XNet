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
	std::vector<Layer*> layers; // list of layers
	float* data; // input on device
	float* h_data; // input on host
	int data_dim; // dimension of one input
	float* label; // label on device
	float* h_label; // label on host
	int label_dim; // dimension of one label (usually 1)
	int size, batch; // whole size of data, batch size
public:
	Network(float* data, int data_dim, float* label, int label_dim,
			int count, int _batch);
	virtual ~Network();
	void Train(int iteration, float alpha); // train the network
	void PushInput(int c, int h, int w);
	void PushOutput(int label_dim);
	void PushConvolution(int c, int kernel);
	void PushPooling(int size, int stride);
	void PushReLU(int output_size);
	void PushSoftmax(int output_size);
	void Pop(); // remove last layer

	void SwitchData(float* h_data, float* h_label, int count);
	void Test(float* h_label, int count);

	void PrintGeneral(); // print general information about layers
	void PrintData(int offset, int r, int c, int precision); // print a datum
};

} /* namespace model */
#endif /* NETWORK_H_ */

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
#include <cmath>

#include "../layer/layer.cuh"
#include "../layer/input.cuh"
#include "../layer/output.cuh"
#include "../layer/convolution.cuh"
#include "../layer/pooling.cuh"
#include "../layer/activation.h"
#include "../layer/relu.cuh"
#include "../layer/softmax.cuh"

#include "../utils/read_data.h"
#include "../utils/write_data.h"

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
	int size, val_size, batch; // whole size of data, batch size
public:
	Network(float* data, int data_dim, float* label, int label_dim,
			int count, int val_size, int _batch);
	virtual ~Network();
	void Train(int iteration, bool debug = false); // train the network
	void PushInput(int c, int h, int w);
	void PushOutput(int label_dim);
	void PushConvolution(int c, int kernel, float alpha);
	void PushPooling(int size, int stride);
	void PushActivation(cudnnActivationMode_t mode);
	void PushReLU(int output_size, float dropout_rate, float alpha);
	void PushSoftmax(int output_size, float dropout_rate, float alpha);
	void Pop(); // remove last layer

	void SwitchData(float* h_data, float* h_label, int count);
	void Test(float* h_label);

	void PrintGeneral(); // print general information about layers
	void PrintData(int offset, int r, int c, int precision); // print a datum

	void ReadParams(std::string dir);
	void SaveParams(std::string dir);
};

} /* namespace model */
#endif /* NETWORK_H_ */

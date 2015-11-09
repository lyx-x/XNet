/*
 * network.h
 *
 * This class uses all layer class to construct a neural network, user should
 * have his data and label array in the host machine (and test data eventually).
 * By calling the constructor and layer inserter, user can build his own network
 * with adjusted parameters like layer size, learning rate, etc..
 *
 * The learning method is gradient descent and the learning rate decreases when
 * validation error stabilizes.
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

#include "../layer/layer.h"
#include "../layer/input.h"
#include "../layer/output.h"
#include "../layer/convolution.h"
#include "../layer/pooling.h"
#include "../layer/activation.h"
#include "../layer/relu.h"
#include "../layer/softmax.h"

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

	float train_error; // previous error rate
	float val_error;
	float lambda; // cumulative learning rate adjustment
public:
	/*
	 * Constructor
	 *
	 * data: pointer to host data array (include training and validation data)
	 * data_dim: dimension of one single datum
	 * label: pointer to host label array
	 * label_dim: dimension of label, usually 1
	 * train_size: training data count
	 * val_size: validation data count
	 * batch_size: batch size, common for all layers
	 */
	Network(float* data, int data_dim, float* label, int label_dim,
			int train_size, int val_size, int batch_size);

	virtual ~Network();

	/*
	 * Train the network: the start may be slow, need to change sigma of initial
	 * weight or adjust learning rate, etc.
	 *
	 * iteration: number of epoch
	 * half_time: threshold for changing the learning rate
	 * half_rate: learning rate adjustment
	 * step_decrease: decrease learning rate by each batch
	 * debug: debug mode (print some extra information)
	 */
	void Train(int iteration, float half_time = 0, float half_rate = 0.5,
			float step_decrease = 1, bool debug = false);

	/*
	 * Insert input layer (should be called at first)
	 *
	 * c: channel number
	 * h: height
	 * w: width
	 */
	void PushInput(int c, int h, int w);

	/*
	 * Insert output layer (should be called in the end)
	 *
	 * label_count: label dimension (example: 10 for digits)
	 */
	void PushOutput(int label_count);

	/*
	 * Insert convolutional layer without activation layer (mode NCHW)
	 *
	 * c: channel of the layer
	 * kernel: square kernel size
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	void PushConvolution(int c, int kernel, float alpha, float sigma = 0.01f,
			float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Insert max pooling layer
	 *
	 * size: size of the square pool
	 * stride: overlapping
	 */
	void PushPooling(int size, int stride);

	/*
	 * Insert activation layer (after convolutional layer)
	 *
	 * mode: activation function (CuDNN constant)
	 */
	void PushActivation(cudnnActivationMode_t mode);

	/*
	 * Insert fully-connected layer with ReLU as activation function
	 *
	 * output_size: output size of the current layer
	 * dropout_rate: rate of dropout
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	void PushReLU(int output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Insert fully-connected layer with Softmax as activation function
	 *
	 * output_size: output size of the current layer
	 * dropout_rate: rate of dropout
	 * alpha: initial learning rate
	 * sigma: initial weight distribution
	 * momentum: momentum of gradient when learning
	 * weight_decay: decay rate of parameters
	 */
	void PushSoftmax(int output_size, float dropout_rate, float alpha,
			float sigma = 0.01f, float momentum = 0.9f, float weight_decay = 0);

	/*
	 * Remove last layer
	 */
	void Pop();

	/*
	 * Switch the host data, used when data size is huge or when testing
	 *
	 * h_data: pointer to new data array
	 * h_label: pointer to new label array
	 * size: data size
	 */
	void SwitchData(float* h_data, float* h_label, int size);

	/*
	 * Test the network, used after switching the test data
	 *
	 * h_label: label array to store the prediction
	 */
	void Test(float* h_label);

	/*
	 * Print general information about layers
	 */
	void PrintGeneral();

	/*
	 * Print a datum
	 *
	 * offset: position of datum
	 * r: row
	 * c: column
	 */
	void PrintData(int offset, int r, int c, int precision);

	/*
	 * Get parameters from files (one folder with multiple files), the actual
	 * network structure should match the saved one
	 *
	 * dir: path to the folder
	 */
	void ReadParams(std::string dir);

	/*
	 * Save parameters to files (one folder with multiple files)
	 *
	 * dir: path to the folder
	 */
	void SaveParams(std::string dir);
};

} /* namespace model */
#endif /* NETWORK_H_ */

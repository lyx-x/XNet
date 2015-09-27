/*
 * network.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: lyx
 */

#include "network.h"

using namespace layer;

namespace model {

Network::Network(float* _data, int _data_dim, float* _label, int _label_dim,
		int count, int batch) {
	this->h_data = _data;
	this->h_label = _label;
	this->size = count;
	this->batch = batch;
	this->data_dim = _data_dim;
	this->label_dim = _label_dim;
	callCuda(cudaMalloc(&data, sizeof(float) * data_dim * size));
	callCuda(cudaMemcpy(data, h_data, sizeof(float) * data_dim * size,
			cudaMemcpyHostToDevice));
	callCuda(cudaMalloc(&label, sizeof(float) * label_dim * size));
	callCuda(cudaMemcpy(label, h_label, sizeof(float) * label_dim * size,
			cudaMemcpyHostToDevice));
}

Network::~Network() {
	h_data = NULL;
	h_label = NULL;
	callCuda(cudaFree(data));
	callCuda(cudaFree(label));
	for (Layer* l : layers)
		delete l;
}

void Network::Train(int iteration, float alpha) {
	// train the network multiple times
	for (int k = 0; k < iteration; k++) {
		// divide the training set to small pieces
		std::cout << "Iteration " << k + 1 << std::endl;
		for (int b = 0; b < size / batch; b++) {
			// choose a new piece and its labels
			layers[0]->data = data + b * batch * data_dim;
			dynamic_cast<Output*>(layers[layers.size() - 1])->label = label + b * batch * label_dim;
			// forward propagation
			for (int i = 0; i < layers.size() - 1; i++)
				layers[i]->forward();
			//std::cout << h_label[b * batch * label_dim] << std::endl;
			//utils::printGpuMatrix(layers[layers.size() - 2]->data, 10, 1, 10, 4);
			// back propagation
			for (int i = layers.size() - 1; i > 0; i--) {
				layers[i]->backward();
				layers[i]->update(alpha); // update the parameters
			}
		}
	}

}

void Network::PushInput(int c, int h, int w) {
	Input* input = new Input(batch, c, h, w, data);
	layers.push_back(input);
}

void Network::PushOutput(int label_dim) {
	Output* output = new Output(layers.back(), label, label_dim, batch);
	layers.push_back(output);
}

void Network::PushConvolution(int c, int kernel) {
	Convolution* conv = new Convolution(layers.back(), batch, c, kernel);
	layers.push_back(conv);
}

void Network::PushPooling(int size, int stride) {
	Pooling* pool = new Pooling(layers.back(), size, stride);
	layers.push_back(pool);
}

void Network::PushReLU(int output_size) {
	ReLU* relu = new ReLU(layers.back(), output_size);
	layers.push_back(relu);
}

void Network::PushSoftmax(int output_size) {
	Softmax* softmax = new Softmax(layers.back(), output_size);
	layers.push_back(softmax);
}

void Network::Pop() {
	Layer* tmp = layers.back();
	layers.pop_back();
	delete tmp;
	layers.back()->next = NULL;
}

void Network::PrintGeneral() {
	std::cout << "Neural Network" << std::endl;
	std::cout << "Layers: " << layers.size() << std::endl;
	int i = 1;
	for (Layer* l : layers)
		std::cout << " - " << i++ << ' ' << l->data_size << std::endl;
}

void Network::PrintData(int offset, int r, int c, int precision) {
	utils::printGpuMatrix(data + offset, r * c, r, c, precision);
}

} /* namespace model */

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
		int count, int _batch) {
	h_data = _data;
	h_label = _label;
	size = count;
	batch = _batch;
	data_dim = _data_dim;
	label_dim = _label_dim;
	callCuda(cudaMalloc(&data, sizeof(float) * data_dim * size));
	callCuda(cudaMemcpy(data, h_data, sizeof(float) * data_dim * batch,
			cudaMemcpyHostToDevice));
	callCuda(cudaMalloc(&label, sizeof(float) * label_dim * size));
	callCuda(cudaMemcpy(label, h_label, sizeof(float) * label_dim * batch,
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

void Network::Train(int iteration) {
	// train the network multiple times
	for (int k = 0; k < iteration; k++) {
		// divide the training set to small pieces
		int offset = 0;
		std::cout << "Iteration " << k + 1 << std::endl;
		for (int b = 0; b < size / batch; b++) {
			// choose a new piece and its labels
			callCuda(cudaMemcpy(data, h_data + offset * data_dim,
					sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
			callCuda(cudaMemcpy(label, h_label + offset * label_dim,
					sizeof(float) * label_dim * batch, cudaMemcpyHostToDevice));
			// forward propagation
			for (int i = 0; i < layers.size(); i++)
				layers[i]->forward();
			// back propagation
			for (int i = layers.size() - 1; i > 0; i--) {
				layers[i]->backward();
				layers[i]->update(); // update the parameters
			}
			offset += batch;
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

void Network::PushConvolution(int c, int kernel, float alpha) {
	Convolution* conv = new Convolution(layers.back(), batch, c, kernel, alpha);
	layers.push_back(conv);
}

void Network::PushPooling(int size, int stride) {
	Pooling* pool = new Pooling(layers.back(), size, stride);
	layers.push_back(pool);
}

void Network::PushReLU(int output_size, float alpha) {
	ReLU* relu = new ReLU(layers.back(), output_size, alpha);
	layers.push_back(relu);
}

void Network::PushSoftmax(int output_size, float alpha) {
	Softmax* softmax = new Softmax(layers.back(), output_size, alpha);
	layers.push_back(softmax);
}

void Network::Pop() {
	Layer* tmp = layers.back();
	layers.pop_back();
	delete tmp;
	layers.back()->next = NULL;
}

void Network::SwitchData(float* h_data, float* h_label, int count) {
	// switch data without modifying the batch size
	size = count;
	this->h_data = h_data;
	this->h_label = h_label;
}

void Network::Test(float* label) {
	int offset = 0;
	for (int b = 0; b < size / batch; b++) {
		callCuda(cudaMemcpy(data, h_data + offset * data_dim,
				sizeof(float) * data_dim * batch, cudaMemcpyHostToDevice));
		callCuda(cudaMemcpy(this->label, h_label + offset * label_dim,
				sizeof(float) * label_dim * batch, cudaMemcpyHostToDevice));
		for (int i = 0; i < layers.size(); i++)
			layers[i]->forward();
		callCuda(cudaMemcpy(label + offset * label_dim,
				layers[layers.size() - 1]->data,
				sizeof(float) * label_dim * batch, cudaMemcpyDeviceToHost));
		offset += batch;
	}
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

void Network::ReadParams(std::string dir) {
	for (int i = 1; i < layers.size() - 1; i++) {
		if (layers[i]->param_size > 0)
			utils::readGPUMatrix(dir + std::to_string(i), layers[i]->param, layers[i]->param_size);
		if (layers[i]->param_bias_size > 0)
			utils::readGPUMatrix(dir + std::to_string(i) + "_bias",
					layers[i]->param_bias, layers[i]->param_bias_size);
	}
}

void Network::SaveParams(std::string dir) {
	for (int i = 1; i < layers.size() - 1; i++) {
		if (layers[i]->param_size > 0)
			utils::writeGPUMatrix(dir + std::to_string(i), layers[i]->param, layers[i]->param_size);
		if (layers[i]->param_bias_size > 0)
			utils::writeGPUMatrix(dir + std::to_string(i) + "_bias",
					layers[i]->param_bias, layers[i]->param_bias_size);
	}
}

} /* namespace model */

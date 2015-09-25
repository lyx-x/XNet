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
	this->size = count;
	this->batch = batch;
	this->data_dim = _data_dim;
	this->label_dim = _label_dim;
	callCuda(cudaMalloc(&data, sizeof(float) * data_dim * size));
	callCuda(cudaMemcpy(data, _data, data_dim * size, cudaMemcpyHostToDevice));
	callCuda(cudaMalloc(&label, sizeof(float) * label_dim * batch));
	callCuda(cudaMemcpy(label, _label, label_dim * batch, cudaMemcpyHostToDevice));

}

Network::~Network() {
	callCuda(cudaFree(data));
	callCuda(cudaFree(label));
	for (Layer* l : layers)
		delete l;
}

void Network::Train(int iteration, float alpha) {
	for (int k = 0; k < iteration; k++) {
		std::cout << "Iteration " << k + 1 << std::endl;
		for (int b = 0; b < size / batch; b++) {
			layers.front()->data = data + b * batch * data_dim;
			dynamic_cast<Output*>(layers[layers.size() - 1])->label = label + b * batch * label_dim;
			for (int i = 0; i < layers.size(); i++)
				layers[i]->forward();
			for (int i = layers.size() - 1; i >= 0; i--) {
				layers[i]->backward();
				layers[i]->update(alpha);
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
	for (Layer* l : layers) {
		std::cout << " - " << i++ << ' ' << l->data_size << std::endl;
		utils::printGpuMatrix(l->data, 10, 1, 10, 6);
	}
}

} /* namespace model */

/*
 * network.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: lyx
 */

#include "network.h"

using namespace layer;

namespace model {

Network::Network(float* data, float* label, int size, int batch) {
	this->data = data;
	this->label = label;
	this->size = size;
	this->batch = batch;
}

Network::~Network() {
	data = NULL;
	label = NULL;
	for (Layer* l : layers)
		delete l;
}

void Network::Train(int iteration, float alpha) {
	for (int k = 0; k < iteration; k++) {
		for (int b = 0; b < size / batch; b++) {
			for (int i = 0; i < layers.size(); i++)
				layers[i]->forward();
			for (int i = layers.size() - 1; i >= 0; i--) {
				layers[i]->backward();
				layers[i]->update(alpha);
			}
		}
	}
}

void Network::PushInput(int n, int c, int h, int w, float* data) {
	Input input(n, c, h, w, data);
	layers.push_back(&input);
}

void Network::PushOutput(float* label, int num_label, int batch) {
	Output output(layers.back(), label, num_label, batch);
	layers.push_back(&output);
}

void Network::PushConvolution(int n ,int c, int h, int w, int kernel) {
	Convolution conv(layers.back(), n, c, h, w, kernel);
	layers.push_back(&convolution);
}

void Network::PushPooling(int size, int stride) {
	Pooling pool(layers.back(), size, stride);
	layers.push_back(&pool);
}

void Network::PushReLU(int output_size) {
	ReLU relu(layers.back(), output_size);
	layers.push_back(&relu);
}

void Network::PushSoftmax(int output_size) {
	Softmax softmax(layers.back(), output_size);
	layers.push_back(&softmax);
}

void Network::Pop() {
	Layer* tmp = layers.back();
	layers.pop_back();
	delete tmp;
	layers.back()->next = NULL;
}

} /* namespace model */

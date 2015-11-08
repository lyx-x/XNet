/*
 * imagenet200.cpp
 *
 *  Created on: Oct 15, 2015
 *      Author: lyx
 */

#include "imagenet200.h"

namespace imagenet200 {

const int label_count = 10; // categories count
const int label_dim = 1;
const int total_size = 5000;

// image size
const int channel = 3;
const int width = 64, height = 64;
const int data_dim = width * height * channel;

string root_path = "../Data/ImageNet200/train/";

int train() {

	int train_size = total_size - 1000;
	int val_size = total_size - train_size;

	// read data

	uint8_t* train_images = new uint8_t[total_size * data_dim];
	ifstream in_images(root_path + "images.bin");
	utils::readBytes(in_images, train_images, total_size * data_dim);
	in_images.close();

	uint8_t* train_labels = new uint8_t[total_size * label_dim];
	ifstream in_labels(root_path + "labels.bin");
	utils::readBytes(in_labels, train_labels, total_size * label_dim);
	in_labels.close();

	float* h_train_images = new float[total_size * data_dim];
	float* h_train_labels = new float[total_size * label_dim];

	for (int i = 0; i < total_size * data_dim; i++)
		h_train_images[i] = (float)train_images[i] / 255.0f;
	for (int i = 0; i < total_size * label_dim; i++)
		h_train_labels[i] = (float)train_labels[i];

	cout << "Training data loaded." << endl;

	// train network

	int batch_size = 50;
	int iteration = 50;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			train_size, val_size, batch_size);
	network.PushInput(channel, height, width); // 3 64 64
	network.PushConvolution(64, 5, -12e-2f, 0.015f, 0.9f, 0.00005f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(64, 5, -12e-2f, 0.015f, 0.9f, 0.00005f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(128, 3, -12e-2f, 0.015f, 0.9f, 0.00005f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(800, 0.5, -10e-2f, 0.015f, 0.9f, 0.00005f);
	network.PushSoftmax(label_count, 0.25, -10e-2f, 0.015f, 0.9f, 0.00005f);
	network.PushOutput(label_count);
	network.PrintGeneral();

	network.Train(iteration, 0, 0.8);

	delete[] train_images;
	delete[] train_labels;

	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

}

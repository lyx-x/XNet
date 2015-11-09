/*
 * imagenet200.cpp
 *
 *  Created on: Oct 15, 2015
 *      Author: lyx
 */

#include "imagenet200.h"

namespace imagenet200 {

const int label_count = 10; // categories count
const int images_per_category = 500 * 9;
const int label_dim = 1;
const int total_size = label_count * images_per_category;

// image size
const int channel = 3;
const int width = 56, height = 56;
const int data_dim = width * height * channel;

string root_path = "../Data/ImageNet200/train/";

int train() {

	int train_size = total_size - label_count * 200;
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

	//utils::showImage(h_train_images + 56 * 56 * 3 * i, 56 ,56 ,3);

	cout << "Training data loaded." << endl;

	// train network

	int batch_size = 50;
	int iteration = 24;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			train_size, val_size, batch_size);
	network.PushInput(channel, height, width);
	network.PushConvolution(48, 5, -6e-2f, 0.02f, 0.9f, 0.001f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(64, 5, -6e-2f, 0.02f, 0.9f, 0.001f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(64, 4, -6e-2f, 0.02f, 0.9f, 0.001f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(800, 0.5, -4e-2f, 0.02f, 0.9f, 0.001f);
	network.PushSoftmax(label_count, 0.25, -4e-2f, 0.02f, 0.9f, 0.001f);
	network.PushOutput(label_count);
	network.PrintGeneral();

	network.Train(iteration, 0, 0.2, 0.94, false);

	delete[] train_images;
	delete[] train_labels;

	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

}

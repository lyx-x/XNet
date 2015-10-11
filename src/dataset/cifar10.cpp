/*
 * cifar10.cpp
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#include "cifar10.h"

namespace cifar10 {

string cifar10_file = "params/cifar10/";

int train() {
	string dir = "../Data/CIFAR10/data_batch_";
	string extension = ".bin";
	string test_file = "../Data/CIFAR10/test_batch.bin";

	int channels = 3;
	int width = 32, height = 32;
	int train_size = 5e4;
	int test_size = 1e4;

	int data_dim = channels * width * height;
	int label_dim = 1;

	uint8_t* train = new uint8_t[train_size * (data_dim + label_dim)];
	for (int i = 1; i <= 5; i++) {
		ifstream in(dir + to_string(i) + extension);
		utils::readBytes(in, train + (i - 1) * (data_dim + label_dim) * 10000,
				(data_dim + label_dim) * 10000);
		in.close();
	}

	float* h_train_images = new float[train_size * data_dim];
	float* h_train_labels = new float[train_size * label_dim];

	int offset = 0;
	for (int i = 0; i < train_size; i++) {
		h_train_labels[i] = (float)train[offset];
		for (int j = 0; j < data_dim; j++)
			h_train_images[i * data_dim + j] = (float)train[offset + 1 + j] / 255.0f;
		offset += data_dim + label_dim;
	}

	int validation_size = 1e4;
	int count = train_size - validation_size; // data size
	int batch_size = 50;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			count, validation_size, batch_size);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(32, 5, -24e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(48, 5, -24e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	//network.PushConvolution(48, 3, -18e-3f);
	//network.PushPooling(2, 2);
	network.PushReLU(64, 0., -18e-3f);
	network.PushSoftmax(10, 0., -15e-3f);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 24;
	cout << "Train " << iteration << " times ..." << endl;
	network.Train(iteration, 0.96);
	network.SaveParams(cifar10_file);
	cout << "End of training ..." << endl;

	uint8_t* test = new uint8_t[test_size * (data_dim + label_dim)];
	ifstream in_test(test_file);
	utils::readBytes(in_test, test, (data_dim + label_dim) * test_size);
	in_test.close();

	float* h_test_images = new float[test_size * data_dim];
	float* h_test_labels = new float[test_size * label_dim];

	offset = 0;
	for (int i = 0; i < test_size; i++) {
		h_test_labels[i] = (float)test[offset];
		for (int j = 0; j < data_dim; j++)
			h_test_images[i * data_dim + j] = (float)test[offset + 1 + j] / 255.0f;
		offset += data_dim + label_dim;
	}

	// test the model
	network.SwitchData(h_test_images, h_test_labels, test_size);

	cout << "Testing ..." << endl;
	float* h_test_labels_predict = new float[test_size];
	network.Test(h_test_labels_predict);
	cout << "End of testing ..." << endl;
	vector<int> errors;
	for (int i = 0; i < test_size; i++) {
		if (abs(h_test_labels_predict[i] - h_test_labels[i]) > 0.1) {
			errors.push_back(i);
			//cout << h_test_labels_predict[i] << ' ' << h_test_labels[i] << endl;
		}
	}
	cout << "Error rate: " << (0.0 + errors.size()) / test_size * 100 << endl;

	delete[] h_test_labels_predict;

	delete[] h_test_images;
	delete[] h_test_labels;
	delete[] test;

	delete[] h_train_images;
	delete[] h_train_labels;
	delete[] train;

	return 0;
}

}

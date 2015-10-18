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
	int total_size = 5e4;
	int test_size = 1e4;

	int data_dim = channels * width * height;
	int label_dim = 1;

	uint8_t* train = new uint8_t[total_size * (data_dim + label_dim)];
	for (int i = 1; i <= 5; i++) {
		ifstream in(dir + to_string(i) + extension);
		utils::readBytes(in, train + (i - 1) * (data_dim + label_dim) * 10000,
				(data_dim + label_dim) * 10000);
		in.close();
	}

	int validation_size = 1e4;
	int train_size = total_size * 2 - validation_size; // data size
	int batch_size = 50;

	// use flip as data augmentation
	float* h_train_images = new float[total_size * data_dim * 2];
	float* h_train_labels = new float[total_size * label_dim * 2];

	int offset = 0;
	int augmentation = total_size * data_dim;
	for (int i = 0; i < total_size; i++) {
		h_train_labels[i] = (float)train[offset];
		h_train_labels[total_size + i] = (float)train[offset];
		for (int j = 0; j < data_dim; j++)
			h_train_images[i * data_dim + j] = (float)train[offset + 1 + j] / 255.0f;
		utils::flipImage(h_train_images + i * data_dim,
				h_train_images + i * data_dim + augmentation, width, height, channels);
		offset += data_dim + label_dim;
	}
	//utils::showImage(h_train_images + 32 * 32 * 3 * 100, 32 ,32 ,3);
	//utils::showImage(h_train_images + 32 * 32 * 3 * 50100, 32 ,32 ,3);
	//utils::showImage(h_train_images + 32 * 32 * 3 * 85000, 32 ,32 ,3);

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			train_size, validation_size, batch_size);
	network.PushInput(channels, height, width); // 3 32 32
	network.PushConvolution(48, 5, -72e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(32, 3, -72e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(800, 0.5, -36e-2f);
	network.PushSoftmax(10, 0.25, -36e-2f);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 24;
	cout << "Train " << iteration << " times ..." << endl;
	network.Train(iteration, 0.95);
	//network.SaveParams(cifar10_file);
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

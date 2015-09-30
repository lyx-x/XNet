/*
 * xnet.cpp
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "model/network.h"
#include "utils/read_data.h"

using namespace std;

int mnist() {
	string train_images_path = "data/MNIST/train-images.idx3-ubyte";
	string train_labels_path = "data/MNIST/train-labels.idx1-ubyte";
	string test_images_path = "data/MNIST/t10k-images.idx3-ubyte";
	string test_labels_path = "data/MNIST/t10k-labels.idx1-ubyte";

	int channels = 1;
	int width, height;
	int train_size, test_size;

	std::cout << "Reading input data" << std::endl;

	// read train data
	ifstream train_images_file(train_images_path, ios::binary);
	train_images_file.seekg(4);
	utils::readInt(train_images_file, &train_size);
	utils::readInt(train_images_file, &height);
	utils::readInt(train_images_file, &width);
	uint8_t* train_images = new uint8_t[train_size * channels * height * width];
	train_images_file.read((char*)train_images, train_size * channels * height * width);

	// read train label
	ifstream train_labels_file(train_labels_path, ios::binary);
	train_labels_file.seekg(8);
	uint8_t* train_labels = new uint8_t[train_size];
	train_labels_file.read((char*)train_labels, train_size);

	std::cout << "Done. Training dataset size: " << train_size << std::endl;

	// transform data
	float* h_train_images = new float[train_size * channels * height * width];
	float* h_train_labels = new float[train_size];
	for (int i = 0; i < train_size * channels * height * width; i++)
		h_train_images[i] = (float)train_images[i] / 255.0f;
	for (int i = 0; i < train_size; i++)
		h_train_labels[i] = (float)train_labels[i];

	// build LeNet
	int data_dim = width * height * channels;
	int label_dim = 1; // 1 column per label
	int count = train_size; // data size
	int batch_size = 200;
	std::cout << "Batch size: " << batch_size << std::endl;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			count, batch_size);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(20, 5);
	network.PushPooling(2, 2);
	network.PushConvolution(50, 5);
	network.PushPooling(2, 2);
	network.PushReLU(200);
	network.PushSoftmax(10);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 10;
	std::cout << "Train " << iteration << " times ..." << std::endl;
	network.Train(iteration, -1e-2f * 100 / batch_size);
	std::cout << "End of training ..." << std::endl;

	// read test cases
	std::cout << "Reading test data" << std::endl;

	ifstream test_images_file(test_images_path, ios::binary);
	test_images_file.seekg(4);
	utils::readInt(test_images_file, &test_size);
	utils::readInt(test_images_file, &height);
	utils::readInt(test_images_file, &width);
	uint8_t* test_images = new uint8_t[test_size * channels * height * width];
	test_images_file.read((char*)test_images, test_size * channels * height * width);

	ifstream test_labels_file(test_labels_path, ios::binary);
	test_labels_file.seekg(8);
	uint8_t* test_labels = new uint8_t[test_size];
	test_labels_file.read((char*)test_labels, train_size);

	std::cout << "Done. Test dataset size: " << test_size << std::endl;

	// transform test data
	float* h_test_images = new float[test_size * channels * height * width];
	float* h_test_labels = new float[test_size];
	for (int i = 0; i < test_size * channels * height * width; i++)
		h_test_images[i] = (float)test_images[i] / 255.0f;
	for (int i = 0; i < test_size; i++)
		h_test_labels[i] = (float)test_labels[i];

	// test the model
	network.SwitchData(h_test_images, h_test_labels, test_size);

	std::cout << "Testing ..." << std::endl;
	float* h_test_labels_predict = new float[test_size];
	network.Test(h_test_labels_predict, test_size);
	std::cout << "End of testing ..." << std::endl;
	vector<int> errors;
	for (int i = 0; i < test_size; i++) {
		if (std::abs(h_test_labels_predict[i] - h_test_labels[i]) > 0.1) {
			errors.push_back(i);
			//std::cout << h_test_labels_predict[i] << ' ' << h_test_labels[i] << endl;
		}
	}
	std::cout << "Error rate: " << (0.0 + errors.size()) / test_size * 100 << std::endl;

	delete[] h_test_labels_predict;
	delete[] test_images;
	delete[] test_labels;
	delete[] h_test_images;
	delete[] h_test_labels;


	delete[] train_images;
	delete[] train_labels;
	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

int cifar10() {
	return 0;
}

int main() {
	std::cout << "XNet v1.0" << std::endl;
	callCuda(cublasCreate(&global::cublasHandle));
	callCudnn(cudnnCreate(&global::cudnnHandle));
	mnist();
	callCuda(cublasDestroy(global::cublasHandle));
	callCudnn(cudnnDestroy(global::cudnnHandle));
	return 0;
}



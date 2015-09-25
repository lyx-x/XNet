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

using namespace std;

string train_images_path = "data/MNIST/train-images.idx3-ubyte";
string train_labels_path = "data/MNIST/train-labels.idx1-ubyte";
string test_images_path = "data/MNIST/t10k-images.idx3-ubyte";
string test_labels_path = "data/MNIST/t10k-labels.idx1-ubyte";
int batch_size = 50;
int iteration = 1;

int lenet() {
	int width = 28;
	int height = 28;
	int channels = 1;

	// Open input data
	std::cout << "Reading input data\n";

	int train_size = 60000;
	int test_size = 10000;

	ifstream train_images_file(train_images_path, ios::binary);
	train_images_file.seekg(16);
	uint8_t* train_images = new uint8_t[train_size * channels * height * width];
	train_images_file.read((char*)train_images, train_size * channels * height * width);

	ifstream train_labels_file(train_labels_path, ios::binary);
	train_labels_file.seekg(8);
	uint8_t* train_labels = new uint8_t[train_size];
	train_labels_file.read((char*)train_labels, train_size);

	printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
	printf("Batch size: %d, iterations: %d\n", batch_size, iteration);

	// transform data
	float* h_train_images = new float[train_size * channels * height * width];
	float* h_train_labels = new float[train_size];
	for (int i = 0; i < train_size * channels * height * width; i++)
		h_train_images[i] = (float)train_images[i];
	for (int i = 0; i < train_size; i++)
		h_train_labels[i] = (float)train_labels[i];

	delete[] train_images;
	delete[] train_labels;

	//utils::printCpuMatrix(train, 28, 1, 28, 6);

	int data_dim = width * height * channels;
	int label_dim = 1;
	int count = train_size;
	/*
	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			count, batch_size);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(20, 5);
	network.PushPooling(2, 2);
	network.PushConvolution(50, 5);
	network.PushPooling(2, 2);
	network.PushReLU(500);
	network.PushSoftmax(10);
	network.PushOutput(10);
	network.PrintGeneral();
	std::cout << "Training ..." << std::endl;
	//network.Train(iteration, 0.001);
	std::cout << "End of training ..." << std::endl;
*/
	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

int main() {
	std::cout << "XNet v1.0" << std::endl;
	callCuda(cublasCreate(&global::cublasHandle));
	callCudnn(cudnnCreate(&global::cudnnHandle));
	lenet();
	callCuda(cublasDestroy(global::cublasHandle));
	callCudnn(cudnnDestroy(global::cudnnHandle));
	return 0;
}



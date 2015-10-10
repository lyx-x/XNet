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
#include "test/test.h"

#include "opencv2/opencv.hpp"

using namespace cv;

using namespace std;

string mnist_file = "params/mnist/";
string cifar10_file = "params/cifar10/";

int train_mnist() {
	string train_images_path = "data/MNIST/train-images.idx3-ubyte";
	string train_labels_path = "data/MNIST/train-labels.idx1-ubyte";
	string test_images_path = "data/MNIST/t10k-images.idx3-ubyte";
	string test_labels_path = "data/MNIST/t10k-labels.idx1-ubyte";

	int channels = 1;
	int width, height;
	int train_size, test_size;

	cout << "Reading input data" << endl;

	// read train data
	ifstream train_images_file(train_images_path, ios::binary);
	train_images_file.seekg(4);
	utils::readInt(train_images_file, &train_size);
	utils::readInt(train_images_file, &height);
	utils::readInt(train_images_file, &width);
	uint8_t* train_images = new uint8_t[train_size * channels * height * width];
	utils::readBytes(train_images_file, train_images,
			train_size * channels * height * width);
	train_images_file.close();

	// read train label
	ifstream train_labels_file(train_labels_path, ios::binary);
	train_labels_file.seekg(8);
	uint8_t* train_labels = new uint8_t[train_size];
	utils::readBytes(train_labels_file, train_labels, train_size);
	train_labels_file.close();

	cout << "Done. Training dataset size: " << train_size << endl;

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
	int val_size = 10000;
	int count = train_size - val_size; // data size
	int batch_size = 50;
	cout << "Batch size: " << batch_size << endl;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			count, val_size, batch_size);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(20, 5, -12e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(50, 5, -12e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(800, 0.5, -10e-3f);
	network.PushSoftmax(10, 0.25, -10e-3f);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 50;
	cout << "Train " << iteration << " times ..." << endl;
	//network.ReadParams(mnist_file);
	network.Train(iteration, 0.96);
	cout << "End of training ..." << endl;

	network.SaveParams(mnist_file);

	// read test cases
	cout << "Reading test data" << endl;

	ifstream test_images_file(test_images_path, ios::binary);
	test_images_file.seekg(4);
	utils::readInt(test_images_file, &test_size);
	utils::readInt(test_images_file, &height);
	utils::readInt(test_images_file, &width);
	uint8_t* test_images = new uint8_t[test_size * channels * height * width];
	test_images_file.read((char*)test_images, test_size * channels * height * width);
	test_images_file.close();

	ifstream test_labels_file(test_labels_path, ios::binary);
	test_labels_file.seekg(8);
	uint8_t* test_labels = new uint8_t[test_size];
	test_labels_file.read((char*)test_labels, test_size);
	test_labels_file.close();

	cout << "Done. Test dataset size: " << test_size << endl;

	// transform test data
	float* h_test_images = new float[test_size * channels * height * width];
	float* h_test_labels = new float[test_size];
	for (int i = 0; i < test_size * channels * height * width; i++)
		h_test_images[i] = (float)test_images[i] / 255.0f;
	for (int i = 0; i < test_size; i++)
		h_test_labels[i] = (float)test_labels[i];

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

void camera_mnist() {
	int channels = 1;
	int width = 28, height = 28;

	float* h_image = new float[channels * height * width];
	float* h_label_predict = new float[1];

	int data_dim = width * height * channels;
	model::Network network(h_image, data_dim, h_label_predict, 1, 1, 0, 1);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(20, 5, -10e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(50, 5, -10e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(400, 0.5, -10e-3f);
	network.PushSoftmax(10, 0.5, -10e-3f);
	network.PushOutput(10);
	network.ReadParams(mnist_file);

	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return;

	Mat image;
	namedWindow("number", 1);
	int threshold = 128;
	createTrackbar("Threshold", "number", &threshold, 255);
	while (true) {
		Mat frame;
		cap >> frame; // get a new frame from camera
		Mat tmp;
		resize(frame, tmp, Size(28, 28));
		cvtColor(tmp, image, CV_BGR2GRAY);
		for (int i = 0; i < channels * height * width; i++) {
			unsigned char _p = image.at<uchar>(i / 28, i % 28);
			image.at<uchar>(i / 28, i % 28) = _p < threshold ? 255 : 0;
		}
		imshow("number", image);
		for (int i = 0; i < channels * height * width; i++) {
			h_image[i] = image.at<uchar>(i / 28, i % 28) / 255.0f;
		}
		network.Test(h_label_predict);
		cout << h_label_predict[0] << endl;
		if (waitKey(100) >= 0)
			break;
	}

	delete[] h_label_predict;
	delete[] h_image;

}

int train_cifar10() {
	string dir = "data/CIFAR10/data_batch_";
	string extension = ".bin";
	string test_file = "data/CIFAR10/test_batch.bin";

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

	int count = train_size; // data size
	int batch_size = 50;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			count, 0, batch_size);
	network.PushInput(channels, height, width); // 1 28 28
	network.PushConvolution(32, 5, -24e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(48, 5, -24e-3f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	//network.PushConvolution(48, 3, -18e-3f);
	//network.PushPooling(2, 2);
	network.PushReLU(64, 0.5, -18e-3f);
	network.PushSoftmax(10, 0.5, -15e-3f);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 24;
	cout << "Train " << iteration << " times ..." << endl;
	network.Train(iteration, false);
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

int main() {
	cout << "XNet v1.0" << endl;
	callCuda(cublasCreate(&global::cublasHandle));
	callCudnn(cudnnCreate(&global::cudnnHandle));

	train_mnist();
	//camera_mnist();
	//train_cifar10();

	callCuda(cublasDestroy(global::cublasHandle));
	callCudnn(cudnnDestroy(global::cudnnHandle));
	cout << "End" << endl;
	return 0;
}



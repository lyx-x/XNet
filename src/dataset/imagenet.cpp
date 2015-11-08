/*
 * imagenet.cpp
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#include "imagenet.h"

namespace imagenet {

struct Image {
	int number;
	int label;
	Image(): number(0), label(0) {}
	Image(int n, int l): number(n), label(l) {}
};

string to_path(Image& image) {
	string label = to_string(image.label);
	while (label.length() < 8)
		label = "0" + label;
	string number = to_string(image.number);
	return "../Data/ImageNet/n" + label + "/n" + label + "_" + number + ".JPEG";
}

int label_count = 5; // use a small part of ImageNet
string label_file_path = "utils/ImageNet/cat" + to_string(label_count) + ".file";

vector<Image> images; // image list
map<int, int> label; // real label to scaled label, from 0 to label_dim - 1

// read all image informations: folder and file name / number
void prepare() {
	label.clear();
	images.clear();

	ifstream label_file(label_file_path);
	int n, m;
	int index = 0;
	while (true) {
		label_file >> n >> m;
		if (n == 0 && m == 0) // EOF
			break;
		images.push_back(Image(m, n));
		if (label.find(n) == label.end())
			label[n] = index++;
	}
	label_file.close();

	assert(label.size() == label_count);
}

int train() {
	srand(time(NULL));

	int batch_size = 2e1;
	int iteration_size = 1e4; // 50 batches in one iteration
	int iteration = 50; // 5e5 images

	// predefined dimensions
	int width = 256;
	int height = 256;
	int channel = 3;
	int data_dim = width * height * channel;
	int label_dim = 1;

	prepare();

	iteration = 1;
	iteration_size = 100;
	batch_size = 1;

	float* h_train_images = new float[iteration_size * data_dim];
	float* h_train_labels = new float[iteration_size * label_dim];

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			iteration_size, 0, batch_size);
	network.PushInput(channel, height, width); // 3 256 256
	network.PushConvolution(48, 11, -20e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(128, 5, -20e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(192, 3, -20e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushConvolution(192, 3, -20e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushConvolution(128, 3, -20e-2f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(2048, 0, -20e-2f);
	network.PushReLU(2048, 0, -20e-2f);
	network.PushSoftmax(label_count, 0, -20e-2f);
	network.PushOutput(label_count);
	network.PrintGeneral();

	for (int i = 0; i < iteration; i++) {
		// load iteration_size images randomly
		int r;
		int offset = 0;
		for (int j = 0; j < iteration_size; j++) {
			r = rand() % images.size();
			Mat im = imread(to_path(images[r]));
			assert(data_dim == im.cols * im.rows * im.elemSize());
			for (int k = 0; k < width * height; k++) {
				for (int c = 0; c < channel; c++)
					h_train_images[k + offset + c * width * height]
					               = ((float)im.at<Vec3b>(k).val[c]) / 255.;
			}
			h_train_labels[j] = label[images[r].label];
			offset += data_dim;
		}

		// train the network several time
		network.Train(10);
	}

	//delete[] h_test_labels_predict;

	//delete[] h_test_images;
	//delete[] h_test_labels;
	//delete[] _test;

	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

}

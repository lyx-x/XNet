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

int label_dim = 2; // use a small part of ImageNet
string label_file_path = "utils/ImageNet/cat" + to_string(label_dim) + ".file";

vector<Image> images;
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

	assert(label.size() == label_dim);
}

int train() {
	int batch_size = 20;

	// predefined dimensions
	int weight = 256;
	int height = 256;
	int channel = 3;
	int data_dim = weight * height * channel;

	prepare();

	for (Image i: images)
		cout << to_path(i) << endl;

	float* h_train_images;
	float* h_train_labels;
	float* _train;
	//delete[] h_test_labels_predict;

	//delete[] h_test_images;
	//delete[] h_test_labels;
	//delete[] _test;

	//delete[] h_train_images;
	//delete[] h_train_labels;
	//delete[] _train;

	return 0;
}

}

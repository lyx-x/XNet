/*
 * imagenet200.cpp
 *
 *  Created on: Oct 15, 2015
 *      Author: lyx
 */

#include "imagenet200.h"

namespace imagenet200 {

int image_per_category = 250;
int label_count = 10;
string* label_string;
int* image_countdown;

string to_train_path(int label, int number = -1) {
	return "../Data/ImageNet200/train/" + label_string[label] + "/images/" +
			label_string[label] + "_" + to_string(number) + ".JPEG";
}

string wnid_path = "../Data/ImageNet200/wnids.txt";

int train() {
	image_countdown = new int[label_count];
	label_string = new string[label_count];

	ifstream wnid_file(wnid_path);
	for (int i = 0; i < label_count; i++) {
		wnid_file >> label_string[i];
		image_countdown[i] = image_per_category - 1;
	}
	wnid_file.close();

	srand(time(NULL));

	int train_size = image_per_category * label_count;
	int batch_size = 50;
	int iteration = 1; // 5e5 images

	// predefined dimensions
	int width = 64;
	int height = 64;
	int channel = 3;
	int data_dim = width * height * channel;
	int label_dim = 1;

	float* h_train_images = new float[train_size * data_dim];
	float* h_train_labels = new float[train_size * label_dim];

	int offset = 0;
	for (int i = 0; i < train_size; i++) {
		int nb_category = rand() % label_count;
		if (image_countdown[nb_category] >= 0) {
			Mat im = imread(to_train_path(nb_category, 250 + image_countdown[nb_category]));
			assert(data_dim == im.cols * im.rows * im.elemSize());
			for (int k = 0; k < width * height; k++) {
				for (int c = 0; c < channel; c++)
					h_train_images[offset + c * width * height + k]
					               = ((float)im.at<Vec3b>(k).val[c]) / 255.;
			}
			h_train_labels[i] = nb_category;
			offset += data_dim;
			image_countdown[nb_category]--;
			im.release();
		}
	}

	cout << "Training data loaded." << endl;

	train_size = 2;
	batch_size = 2;

	model::Network network(h_train_images, data_dim, h_train_labels, label_dim,
			train_size, 0, batch_size);
	network.PushInput(channel, height, width); // 3 64 64
	network.PushConvolution(32, 5, -2e1f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(48, 5, -2e1f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(4, 4);
	network.PushReLU(256, 0, -1e1f);
	network.PushSoftmax(label_count, 0, -1e1f);
	network.PushOutput(label_count);
	network.PrintGeneral();

	network.Train(10, 1, true);

	delete[] h_train_images;
	delete[] h_train_labels;

	delete[] image_countdown;
	delete[] label_string;

	return 0;
}

}

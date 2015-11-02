/*
 * mnist.cpp
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#include "mnist.h"

namespace mnist {

string mnist_file = "params/mnist/";

int train() {
	string train_images_path = "../Data/MNIST/train-images.idx3-ubyte";
	string train_labels_path = "../Data/MNIST/train-labels.idx1-ubyte";
	string test_images_path = "../Data/MNIST/t10k-images.idx3-ubyte";
	string test_labels_path = "../Data/MNIST/t10k-labels.idx1-ubyte";

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

	int data_dim = width * height * channels;
	int label_dim = 1; // 1 column per label
	int val_size = 10000;
	train_size -= val_size; // 50000

	// augment data
	int train_size_augmented = 0;
	int width_augmented = 0;
	int height_augmented = 0;
	int augment_delta = 1;
	utils::getDimension(train_size, height, width, augment_delta, train_size_augmented,
			width_augmented, height_augmented);
	int data_dim_augmented = width_augmented * height_augmented * channels;

	cout << "Augmented training dataset size: " << train_size_augmented << endl;

	float* h_train_images_augmented = new float[(train_size_augmented + val_size) *
	                                            data_dim_augmented];
	float* h_train_labels_augmented = new float[train_size_augmented + val_size];

	utils::translationAugmentation(h_train_images, train_size, width, height,
			augment_delta, h_train_images_augmented, train_size_augmented, channels,
			width_augmented, height_augmented);
	utils::translationAugmentation(h_train_images + train_size * data_dim,
			val_size, width, height, 0,
			h_train_images_augmented + train_size_augmented * data_dim_augmented,
			val_size, channels, width_augmented, height_augmented);

	for (int i = 0; i < train_size_augmented / train_size; i++)
		for (int j = 0; j < train_size; j++)
			h_train_labels_augmented[i * train_size + j] = h_train_labels[j];
	for (int i = 0; i < val_size; i++)
		h_train_labels_augmented[i + train_size_augmented] =
				h_train_labels[i + train_size];

	// view augmentation
	/*
	while (true) {
		int index = rand() % (val_size + train_size_augmented);
		cout << h_train_labels_augmented[index] << endl;
		utils::showImage(h_train_images_augmented,
				width_augmented, height_augmented, channels, index);
	}
	*/

	// build LeNet
	int batch_size = 50;
	cout << "Batch size: " << batch_size << endl;
	/*
	model::Network network(h_train_images_augmented, data_dim_augmented,
			h_train_labels_augmented, label_dim, train_size_augmented,
			val_size, batch_size);
	network.PushInput(channels, height_augmented, width_augmented); // 1 27 27
	*/
	model::Network network(h_train_images, data_dim,
			h_train_labels, label_dim, train_size,
			val_size, batch_size);
	network.PushInput(channels, height, width); // 1 28 28

	network.PushConvolution(20, 5, -10e-2f, 0.01f, 0.9f, 0.00005f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushConvolution(40, 5, -10e-2f, 0.01f, 0.9f, 0.00005f);
	network.PushActivation(CUDNN_ACTIVATION_RELU);
	network.PushPooling(2, 2);
	network.PushReLU(800, 0.6, -10e-2f, 0.01f, 0.9f, 0.00005f);
	network.PushSoftmax(10, 0.25, -10e-2f, 0.01f, 0.9f, 0.00005f);
	network.PushOutput(10);
	network.PrintGeneral();

	// train the model
	int iteration = 50;
	cout << "Train " << iteration << " times ..." << endl;
	//network.ReadParams(mnist_file);
	network.Train(iteration); // depend on the number of batch_size
	cout << "End of training ..." << endl;

	//network.SaveParams(mnist_file);

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

	float* h_test_images_augmented = new float[test_size * data_dim_augmented];
	utils::translationAugmentation(h_test_images, test_size, width, height,
			0, h_test_images_augmented, test_size, channels,
			width_augmented, height_augmented);

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

	delete[] h_test_images_augmented;

	delete[] train_images;
	delete[] train_labels;

	delete[] h_train_images_augmented;
	delete[] h_train_labels_augmented;

	delete[] h_train_images;
	delete[] h_train_labels;

	return 0;
}

void camera() {
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

}

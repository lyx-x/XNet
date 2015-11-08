#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const int width = 64;
const int height = 64;
const int channel = 3;
const int data_dim = width * height * channel;

vector<string> label_string;
vector<int> image_countdown;

const string root = "/home/lyx/workspace/cuda/Data/ImageNet200/";
const string wnid_path = root + "wnids.txt";

string to_train_path(int label, int number = -1) {
    return root + "train/" + label_string[label] + "/images/" +
           label_string[label] + "_" + to_string(number) + ".JPEG";
}

int get_rand(int max) {
    return rand() % max;
}

void save_file(string name, uint8_t* data, int size) {
    ofstream out(name);
    out.write((char*)data, size);
    out.close();
}

void get_train_images(int argc, char** argv) {

    int image_per_category = 500;
    int label_count = 10; // choose some categories

    if (argc >= 3) {
        label_count = atoi(argv[2]);
        image_per_category = atoi(argv[3]);
    }

    int data_size = data_dim * label_count * image_per_category;
    int label_size = label_count * image_per_category;

    ifstream wnid_file(wnid_path);
    string tmp;
    for (int i = 0; i < label_count; i++) {
        wnid_file >> tmp;
        label_string.push_back(tmp);
        image_countdown.push_back(image_per_category - 1);
    }
    wnid_file.close();

    uint8_t* bin_images = new uint8_t[data_size];
    uint8_t* bin_labels = new uint8_t[label_size];

    int index = 0;
    while (!label_string.empty()) {
        uint8_t category = get_rand(label_string.size());
        int image = image_countdown[category];
        Mat m = imread(to_train_path(category, image));
        assert(data_dim == m.cols * m.rows * m.elemSize());
        int _index = index * data_dim;
        for (int i = 0; i < channel; i++)
            for (int j = 0; j < height; j++)
                for (int k = 0; k < width; k++) {
                    Vec3b pixel = m.at<Vec3b>(j, k);
                    bin_images[_index] = (uint8_t)pixel[i];
                    _index++;
                }
        bin_labels[index] = category;
        m.release();
        if (image == 0) {
            label_string.erase(label_string.begin() + category);
            image_countdown.erase(image_countdown.begin() + category);
        }
        else
            image_countdown[category]--;
        index++;
    }

    save_file(root + "train/images.bin", bin_images, data_size);
    save_file(root + "train/labels.bin", bin_labels, label_size);

    delete[] bin_images;
    delete[] bin_labels;

    cout << "Done" << endl;
}

void get_validation_images(int argc, char** argv) {

}

void get_test_images(int argc, char** argv) {

}

/*
 * program 0 10 500
 * program 1 500
 * program 2 500
 */
int main(int argc, char** argv) {
    switch (argv[1][0]) {
        case '0':
            get_train_images(argc, argv);
            break;
        case '1':
            get_validation_images(argc, argv);
            break;
        case '2':
            get_test_images(argc, argv);
            break;
    }
    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

const string mode = "train/";

const int image_per_category = 500;
const int label_count = 10; // choose some categories

const int width = 64;
const int height = 64;
const int channel = 3;
const int data_dim = width * height * channel;

const int data_size = data_dim * label_count * image_per_category;
const int label_size = label_count * image_per_category;

vector<string> label_string;
vector<int> image_countdown;

const string root = "/home/lyx/workspace/cuda/Data/ImageNet200/";
const string wnid_path = root + "wnids.txt";

string to_path(int label, int number = -1) {
    return root + mode + label_string[label] + "/images/" +
           label_string[label] + "_" + to_string(number) + ".JPEG";
}

int main() {

    ifstream wnid_file(wnid_path);
    string tmp;
    for (int i = 0; i < label_count; i++) {
        wnid_file >> tmp;
        label_string.push_back(tmp);
        image_countdown.push_back(image_per_category - 1);
    }
    wnid_file.close();

    cout << "Hello, World!" << endl;


    return 0;
}
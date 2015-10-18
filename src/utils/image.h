/*
 * image.h
 *
 *  Created on: Oct 18, 2015
 *      Author: lyx
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace utils {

void showImage(float* img, int width, int height, int channel);
void flipImage(float* src, float* dest, int width, int height, int channel);

}

#endif /* IMAGE_H_ */

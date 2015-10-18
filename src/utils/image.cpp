/*
 * image.cpp
 *
 *  Created on: Oct 18, 2015
 *      Author: lyx
 */

#include "image.h"

namespace utils {

void showImage(float* img, int width, int height, int channel) {
	Mat I(height, width, CV_8UC3, Scalar::all(0));
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			for (int c = 0; c < channel; c++)
				I.at<Vec3b>(j, i).val[c] =
						uchar(img[width * height * c + width * j + i] * 255);
	imshow("Image", I);
	waitKey(0);
}

void flipImage(float* src, float* dest, int width, int height, int channel) {
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			for (int c = 0; c < channel; c++)
				dest[width * height * c + width * j + i] =
						src[width * height * c + width * j + (width - i - 1)];
}

}

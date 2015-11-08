/*
 * image.cpp
 *
 *  Created on: Oct 18, 2015
 *      Author: lyx
 */

#include "image.h"

namespace utils {

void showImage(float* img, int width, int height, int channel, int offset) {
	img += offset * width * height * channel;
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

void translationAugmentation(float* old_m, int old_n, int old_w, int old_h,
		int delta, float* m, int n, int c, int w, int h) {
	int _k = 0;
	for (int i = 0; i <= delta; i += delta)
		for (int j = 0; j <= delta; j += delta)
			for (int k = 0; k < old_n; k++)
				for (int x = 0; x < c; x++)
					for (int y = 0; y < h; y++)
						for (int z = 0; z < w; z++) {
							m[_k] = old_m[k * c * old_w * old_h + x * old_w * old_h +
							              (y + i) * old_w + (z + j)];
							_k++;
						}
}

}

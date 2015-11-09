/*
 * image.h
 *
 * Image utilities with data augmentation
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

/*
 * Display the image with OpenCV
 *
 * img: pointer to images array
 * width: image width
 * height: image height
 * channel: image channel
 * offset: index of image in the array
 */
void showImage(float* img, int width, int height, int channel, int offset = 0);

/*
 * Flip an image horizontally
 *
 * src: pointer of old image
 * dest: pointer of new image
 * width: image width
 * height: image height
 * channel: image channel
 */
void flipImage(float* src, float* dest, int width, int height, int channel);

/*
 * Translate an image and get 4 cropped images
 *
 * old_m: pointer to old images
 * old_n: old images count
 * old_w: old image width
 * old_h: old image height
 * delta: crop size
 * m: pointer to new images array
 * n: new images count
 * c: image channel (does not change)
 * w: new image width
 * h: new image height
 */
void translationAugmentation(float* old_m, int old_n, int old_w, int old_h,
		int delta, float* m, int n, int c, int w, int h);

}

#endif /* IMAGE_H_ */

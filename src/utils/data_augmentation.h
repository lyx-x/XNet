/*
 * data_augmentation.h
 *
 *  Created on: Nov 1, 2015
 *      Author: lyx
 */

#ifndef DATA_AUGMENTATION_H_
#define DATA_AUGMENTATION_H_

namespace utils {

void getDimension(int old_n, int old_w, int old_h, int delta, int& n, int& w, int& h);

void translationAugmentation(float* old_m, int old_n, int old_w, int old_h,
		int delta, float* m, int n, int c, int w, int h);

}

#endif /* DATA_AUGMENTATION_H_ */

/*
 * data_augmentation.cpp
 *
 *  Created on: Nov 1, 2015
 *      Author: lyx
 */

#include "data_augmentation.h"

namespace utils {

void getDimension(int old_n, int old_w, int old_h, int delta, int& n, int& w, int& h) {
	w = old_w - delta;
	h = old_h - delta;
	n = old_n * (delta + 1) * (delta + 1);
}

void translationAugmentation(float* old_m, int old_n, int old_w, int old_h,
		int delta, float* m, int n, int c, int w, int h) {
	int _k = 0;
	for (int i = 0; i <= delta; i++)
		for (int j = 0; j <= delta; j++)
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

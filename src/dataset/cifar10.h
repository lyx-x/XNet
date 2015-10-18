/*
 * cifar10.h
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#ifndef CIFAR10_H_
#define CIFAR10_H_

#include "../model/network.h"
#include "../utils/read_data.h"
#include "../utils/image.h"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace cifar10 {

int train();

}

#endif /* CIFAR10_H_ */

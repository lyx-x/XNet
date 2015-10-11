/*
 * imagenet.h
 *
 *  Created on: Oct 11, 2015
 *      Author: lyx
 */

#ifndef IMAGENET_H_
#define IMAGENET_H_

#include <vector>
#include <map>
#include <algorithm>
#include <cassert>

#include "../model/network.h"
#include "../utils/read_data.h"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace imagenet {

int train();

}

#endif /* IMAGENET_H_ */

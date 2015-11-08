/*
 * imagenet200.h
 *
 *  Created on: Oct 15, 2015
 *      Author: lyx
 */

#ifndef IMAGENET200_H_
#define IMAGENET200_H_

#include <vector>
#include <map>
#include <string>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cassert>

#include "../model/network.h"
#include "../utils/read_data.h"
#include "../utils/image.h"

using namespace std;

namespace imagenet200 {

int train();

}

#endif /* IMAGENET200_H_ */

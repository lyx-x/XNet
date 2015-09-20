/*
 * set_value.cuh
 *
 *  Created on: Sep 8, 2015
 *      Author: lyx
 */

#ifndef SET_VALUE_CUH_
#define SET_VALUE_CUH_

#include <ctime>
#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

namespace utils {

void setGpuNormalValue(float* x, int n, float mean, float stddev);
void setGpuNormalValue(float* x, int n);
}

#endif /* SET_VALUE_CUH_ */

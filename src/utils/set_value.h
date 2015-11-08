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

#include "utils.h"
#include "global.h"
#include "print.h"

namespace utils {

void setGpuValue(float* x, int n, float val);
void setGpuUniformValue(float* x, int n, int input_size, int output_size);
void setGpuUniformValue(float* x, int n, float epsilon);
void setGpuNormalValue(float* x, int n, float mean, float stddev);
void setGpuNormalValue(float* x, int n, int scale = -1);
void dropGpuValue(float *x, int n, float dropout_rate);
void scaleGpuValue(float *x, int n, float scale);

}

#endif /* SET_VALUE_CUH_ */

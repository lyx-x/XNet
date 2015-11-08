/*
 * set_value.h
 *
 * Utility functions to set or change value to host or device array
 *
 *  Created on: Sep 8, 2015
 *      Author: lyx
 */

#ifndef SET_VALUE_H_
#define SET_VALUE_H_

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

/*
 * Set one single value to all the array
 *
 * x: pointer to array
 * n: array length
 * val: value
 */
void setGpuValue(float* x, int n, float val);

/*
 * Set uniformly distributed value
 *
 * x: pointer to array
 * n: array length
 * epsilon: value interval
 */
void setGpuUniformValue(float* x, int n, float epsilon);

/*
 * Set Gaussian distributed value
 *
 * x: pointer to array
 * n: array length
 * mean: mean of Gaussian distribution
 * stddev: standard deviation of Gaussian distribution
 */
void setGpuNormalValue(float* x, int n, float mean, float stddev);

/*
 * Set part of the array to zero (dropout)
 *
 * x: pointer to array
 * n: array length
 * dropout_rate: proportion of array to be set to 0
 */
void dropGpuValue(float* x, int n, float dropout_rate);

/*
 * Scale the array
 *
 * x: pointer to array
 * n: array length
 * scale: scale rate
 */
void scaleGpuValue(float* x, int n, float scale);

}

#endif /* SET_VALUE_H_ */

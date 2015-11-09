/*
 * print.h
 *
 * Methods to print out host or device array
 *
 *  Created on: Sep 8, 2015
 *      Author: lyx
 */

#ifndef PRINT_H_
#define PRINT_H_

#include <iostream>
#include <iomanip>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "utils.h"
#include "global.h"

namespace utils {

void printCpuMatrix(float* m, int n, int r, int c = 1, int precision = 2);
void printGpuMatrix(float* d_m, int n, int r, int c = 1, int precision = 2);

/*
 * Print the maximum of an array
 *
 * d_m: pointer to array
 * n: array length
 * precision: display precision
 */
void printGpuMax(float* d_m, int n, int precision = 4);

}

#endif /* PRINT_H_ */

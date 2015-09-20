/*
 * print.cuh
 *
 *  Created on: Sep 8, 2015
 *      Author: lyx
 */

#ifndef PRINT
#define PRINT

#include <iostream>
#include <iomanip>

#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace utils {

void printCpuMatrix(float* m, int n, int r, int c = 1, int precision = 2);
void printGpuMatrix(float* d_m, int n, int r, int c = 1, int precision = 2);

}

#endif /* PRINT */

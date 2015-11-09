/*
 * global.h
 *
 * Global instances such as CUDA handle
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include "cublas_v2.h"
#include "cudnn.h"

namespace global {

extern cudnnHandle_t cudnnHandle;
extern cublasHandle_t cublasHandle;

} /* namespace global */
#endif /* GLOBAL_H_ */

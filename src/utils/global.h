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
#include <string>

namespace global {

extern cudnnHandle_t cudnnHandle;
extern cublasHandle_t cublasHandle;
    const std::string root = "/home/lyx/workspace/cuda/XNet/";

} /* namespace global */
#endif /* GLOBAL_H_ */

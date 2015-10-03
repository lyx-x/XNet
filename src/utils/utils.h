/*
 * utils.h
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <sstream>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define callCudnn(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
    	_error << "CuDNN failure: " << cudnnGetErrorString(status);    \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define callCurand(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS) {                             \
    	_error << "CuRAND failure: " << status;   					   \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

#define callCuda(status) do {                                  		   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
    	_error << "Cuda failure: " << status;                          \
    	FatalError(_error.str());                                      \
    }                                                                  \
} while(0)

namespace utils {


} /* namespace utils */
#endif /* UTILS_H_ */

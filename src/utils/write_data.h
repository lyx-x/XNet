/*
 * write_data.h
 *
 *  Created on: Oct 3, 2015
 *      Author: lyx
 */

#ifndef WRITE_DATA_H_
#define WRITE_DATA_H_

#include <fstream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "utils.h"

namespace utils {

void writeGPUMatrix(std::string dir, float* d_val, int n);
void writeCPUMatrix(std::string dir, float* val, int n);

}

#endif /* WRITE_DATA_H_ */

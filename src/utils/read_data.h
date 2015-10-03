/*
 * read_data.h
 *
 *  Created on: Sep 28, 2015
 *      Author: lyx
 */

#ifndef READ_DATA_H_
#define READ_DATA_H_

#include <fstream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace utils {

void readInt(std::ifstream& stream, int* val);
void readBytes(std::ifstream& stream, uint8_t* val, int n);
void readFloat(std::ifstream& stream, float* val, int n);
void readCPUMatrix(std::string dir, float* val, int n);
void readGPUMatrix(std::string dir, float* d_val, int n);

}

#endif /* READ_DATA_H_ */

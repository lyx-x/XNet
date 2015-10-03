/*
 * write_data.cpp
 *
 *  Created on: Oct 3, 2015
 *      Author: lyx
 */

#include "write_data.h"

namespace utils {

void writeGPUMatrix(std::string dir, float* d_val, int n) {
	float* val = new float[n];
	cudaMemcpy(val, d_val, sizeof(float) * n, cudaMemcpyDeviceToHost);
	std::ofstream out(dir);
	out.write((char*)val, sizeof(float) * n);
	out.close();
	delete[] val;
}

void writeCPUMatrix(std::string dir, float* val, int n) {
	std::ofstream out(dir);
	out.write((char*)val, sizeof(float) * n);
	out.close();
}

}

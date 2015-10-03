/*
 * test.cpp
 *
 *  Created on: Oct 3, 2015
 *      Author: lyx
 */

#include "test.h"

namespace test {

void binaryFileTest() {
	float* val = new float[5];
	for (int i = 0; i < 5; i++)
		val[i] = 0.01 * exp(4) / (i * i + sqrt(0.4));
	float* d_val;
	callCuda(cudaMalloc(&d_val, sizeof(float) * 5));
	callCuda(cudaMemcpy(d_val, val, sizeof(float) * 5, cudaMemcpyHostToDevice));
	std::string dir = "src/test/test_file";
	std::ofstream out(dir);
	utils::writeGPUMatrix(dir, d_val, 5);
	float* h_val = new float[5];
	utils::readCPUMatrix(dir, h_val, 5);
	for (int i = 0; i < 5; i++)
		assert(abs(h_val[i] - val[i]) < 1e-5);
	delete[] val;
	delete[] h_val;
	callCuda(cudaFree(d_val));
}

}



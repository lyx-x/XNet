/*
 * print.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "print.h"

namespace utils {

void printCpuMatrix(float* m, int n, int r, int c, int precision) {
	std::cout << "Size: " << r << " * " << c << std::endl;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (i + j * r >= n)
				break;
			std::cout << std::fixed << std::setprecision(precision) << m[i + j * r] << '\t';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void printGpuMatrix(float* d_m, int n, int r, int c, int precision) {
	float* m = new float[n];
	cublasGetVector(n, sizeof(*m), d_m, 1, m, 1);
	std::cout << "Size: " << r << " * " << c << std::endl;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			std::cout << std::fixed << std::setprecision(precision) << m[i + j * r] << '\t';
		std::cout << std::endl;
	}
	std::cout << std::endl;
	delete[] m;
}

void printGpuMax(float* d_m, int n, int precision) {
	int result;
	callCuda(cublasIsamax(global::cublasHandle, n, d_m, 1, &result));
	float ans;
	callCuda(cudaMemcpy(&ans, d_m + result - 1, sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << std::fixed << std::setprecision(precision) <<
			"Max: " << ans << std::endl;

}

}

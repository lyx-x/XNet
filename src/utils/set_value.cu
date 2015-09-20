/*
 * set_value.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "set_value.cuh"

namespace utils {

__global__ void setValue(float* x, int n, float val) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		x[i] = val;
}

void setGpuValue(float* x, int n, float val) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	setValue<<<blocksPerGrid, threadsPerBlock>>>(x, n, val);
}

void setGpuNormalValue(float* x, int n, float mean, float stddev) {
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerateNormal(generator, x, n, mean, stddev);
	curandDestroyGenerator(generator);
}

void setGpuNormalValue(float* x, int n) {
	setGpuNormalValue(x, n, 0, sqrt(3.0f / n));
}

}

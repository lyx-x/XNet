/*
 * set_value.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "set_value.cuh"

using namespace global;

namespace utils {

__global__ void setValue(float* x, int n, float val) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		x[i] = val;
}

__global__ void dropout(float* x, float* t, int n, float threshold) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		if (t[i] < threshold)
			x[i] = 0;
}

__global__ void scale(float* x, int n, float epsilon) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		x[i] = x[i] * epsilon * 2 - epsilon;
}

void setGpuValue(float* x, int n, float val) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	setValue<<<blocksPerGrid, threadsPerBlock>>>(x, n, val);
}

void setGpuUniformValue(float* x, int n, int input_size, int output_size) {
	float epsilon = std::sqrt(6.0 / (input_size + output_size));
	setGpuUniformValue(x, n, epsilon);
}

void setGpuUniformValue(float* x, int n, float epsilon) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerateUniform(generator, x, n);
	scale<<<blocksPerGrid, threadsPerBlock>>>(x, n, epsilon);
	curandDestroyGenerator(generator);
}

void setGpuNormalValue(float* x, int n, float mean, float stddev) {
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerateNormal(generator, x, n, mean, stddev);
	curandDestroyGenerator(generator);
}

void setGpuNormalValue(float* x, int n, int scale) {
	if (scale == -1)
		scale = n;
	setGpuNormalValue(x, n, 0, sqrt(0.3f / scale));
}

void dropGpuValue(float *x, int n, float dropout_rate) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	float* t;
	cudaMalloc((void**)&t, sizeof(float) * n);
	curandGenerateUniform(generator, t, n);
	dropout<<<blocksPerGrid, threadsPerBlock>>>(x, t, n, dropout_rate);
	cudaFree(t);
	curandDestroyGenerator(generator);
}

void scaleGpuValue(float *x, int n, float scale) {
	cublasSscal(cublasHandle, n, &scale, x, 1);
}

}

/*
 * set_value.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "set_value.cuh"

namespace utils {

void setGpuNormalValue(float* x, int n, float mean, float stddev) {
	curandGenerator_t generator;
	callCurand(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
	callCurand(curandSetPseudoRandomGeneratorSeed(generator, time(NULL)));
	callCurand(curandGenerateNormal(generator, x, n, mean, stddev));
	callCurand(curandDestroyGenerator(generator));
}

void setGpuNormalValue(float* x, int n) {
	setGpuNormalValue(x, n, 0, sqrt(3.0 / n));
}

}

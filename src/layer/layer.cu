/*
 * layer.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "layer.cuh"

namespace layer {

Layer::Layer(float alpha, float momentum, float weight_decay):
		alpha(alpha), momentum(momentum), weight_decay(weight_decay) {
	isProduction = false;
}

Layer::~Layer() {}

void Layer::adjust_learning(float scale) {
	alpha *= scale;
}


}


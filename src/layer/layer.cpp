/*
 * layer.cu
 *
 *  Created on: Sep 20, 2015
 *      Author: lyx
 */

#include "layer.h"

namespace layer {

Layer::Layer(float alpha, float momentum, float weight_decay):
		alpha(alpha), momentum(momentum), weight_decay(weight_decay) {

}

Layer::~Layer() {}

void Layer::adjust_learning(float scale) {
	alpha *= scale;
}


}


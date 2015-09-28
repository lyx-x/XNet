/*
 * read_data.cpp
 *
 *  Created on: Sep 28, 2015
 *      Author: lyx
 */

#include "read_data.h"

namespace utils {

void readInt(std::ifstream& stream, int* val) {
	// little endian
	for (int i = sizeof(int) - 1; i >= 0; i--)
		stream.read(((char*)val) + i, 1);
}

}



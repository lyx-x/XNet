/*
 * read_data.h
 *
 *  Created on: Sep 28, 2015
 *      Author: lyx
 */

#ifndef READ_DATA_H_
#define READ_DATA_H_

#include <fstream>

namespace utils {

void readInt(std::ifstream& stream, int* val);

}

#endif /* READ_DATA_H_ */

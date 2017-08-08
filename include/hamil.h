#ifndef HAMIL_H
#define HAMIL_H

#include "linalg.h"
#include "block.h"

MAT_TYPE *HeisenH_int(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2);

MAT_TYPE *HeisenH_int_r(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2,
					const int num_ind, const int *restrict inds);

#endif
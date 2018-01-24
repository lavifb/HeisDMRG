#ifndef HAMIL_H
#define HAMIL_H

#include "linalg.h"
#include "block.h"
#include "model.h"

MAT_TYPE *getLowestEStates(const DMRGBlock *sys_enl, const DMRGBlock *env_enl, const model_t* model, int num_states, MAT_TYPE **psi0_guessp, double *energies);

MAT_TYPE *HeisenH_int(const model_t* model, const DMRGBlock *block1, const DMRGBlock *block2);

MAT_TYPE *HeisenH_int_r(const model_t* model, const DMRGBlock *block1, const DMRGBlock *block2,
	const int num_ind, const int *restrict inds);

hamil_mats_t *HeisenH_int_mats(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2);

void freehamil_mats_t(hamil_mats_t *hamil_mats);

#endif
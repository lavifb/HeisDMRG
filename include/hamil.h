#ifndef HAMIL_H
#define HAMIL_H

#include "linalg.h"
#include "block.h"
#include "model.h"

MAT_TYPE *getLowestEStates(const DMRGBlock *sys_enl, const DMRGBlock *env_enl, const model_t* model, int num_restr_ind,
	const int *restr_basis_inds, int num_states, MAT_TYPE **psi0_guessp, double *energies);

MAT_TYPE *HeisenH_int(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2);

MAT_TYPE *HeisenH_int_r(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2,
	const int num_ind, const int *restrict inds);

Hamil_mats *HeisenH_int_mats(double *H_params, const DMRGBlock *block1, const DMRGBlock *block2);

void freeHamil_mats(Hamil_mats *hamil_mats);

#endif
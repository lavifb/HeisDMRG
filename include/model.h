#ifndef MODEL_H
#define MODEL_H

#include "linalg.h"
#include "block.h"

typedef struct model_t {
	int d_model; // single site basis size
	MAT_TYPE *H1;  // single site Hamiltonian
	MAT_TYPE *Sz;  // single site Sz
	MAT_TYPE *Sp;  // single site S+
	MAT_TYPE *Id;  // single site Identity Matrix
	int *init_mzs;	// 2*mz quantum number for each state
	int num_ops;
	MAT_TYPE **init_ops; // single site block tracked operators

	DMRGBlock *single_block; // block for single site

	int fullLength; // Full length of system
	int ladder_width;
	
	void *H_params;
	// Pointer to interaction Hamiltonian
	MAT_TYPE *(*H_int)(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2);
	// Pointer to interaction Hamiltonian used in DMRG step
	#if USE_PRIMME
	hamil_mats_t *(*H_int_mats)(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2);
	#else
	MAT_TYPE *(*H_int_r)(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2,
		const int num_ind, const int *restrict inds);
	#endif
} model_t;


void compileParams(model_t *model);

model_t *newNullModel();

void freeModel(model_t *model);

model_t *newHeis2Model();

#endif
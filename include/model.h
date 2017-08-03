#ifndef MODEL_H
#define MODEL_H

#include "block.h"

typedef struct model_t {
	int d_model; // single site basis size
	double *H1;  // single site Hamiltonian
	double *Sz;  // single site Sz
	double *Sp;  // single site S+
	double *Id;  // single site Identity Matrix
	int *init_mzs;	// 2*mz quantum number for each state
	int num_ops;
	double **init_ops; // single site block tracked operators

	DMRGBlock *single_block; // block for single site
	
	double J;
	double Jz;
	double *H_params;
	// Pointer to interaction Hamiltonian
	double *(*H_int)(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2);
	double *(*H_int_r)(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2, 
					const int num_ind, const int *restrict inds);
} model_t;


void compileParams(model_t *model);

model_t *newNullModel();

void freeModel(model_t *model);

#endif
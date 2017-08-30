#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>
#include <stdio.h>
#include <string.h>
#include <float.h>


DMRGBlock *createDMRGBlock(model_t *model, int fullLength) {

	DMRGBlock *block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

	block->length = 1;
	block->fullLength = fullLength;
	block->side = 'L';
	block->meas = 'N';

	int dim = model->d_model;
	block->d_block = dim;
	block->num_ops = model->num_ops;
	block->model   = model;

	block->mzs = (int *)mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	memcpy(block->mzs, model->init_mzs, dim * sizeof(int));

	// copy operators
	block->ops = (MAT_TYPE **)mkl_malloc(block->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i = 0; i < block->num_ops; i++) {
		block->ops[i] = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(block->ops[i], model->init_ops[i], dim*dim * sizeof(MAT_TYPE));
	}

	block->psi = (MAT_TYPE *)mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	block->A = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	block->d_trans = 0;
	block->trans = NULL;

	// Set energy to ridiculous value
	block->energy = DBL_MAX;
	block->trunc_err = 0;

	return block;
}

DMRGBlock *copyDMRGBlock(DMRGBlock *orig) {

	DMRGBlock *newBlock = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

	newBlock->length  = orig->length;
	newBlock->fullLength = orig->fullLength;
	newBlock->side  = orig->side;
	newBlock->meas  = orig->meas;
	int dim = orig->d_block;
	newBlock->d_block = dim;
	newBlock->num_ops = orig->num_ops;
	newBlock->model   = orig->model;

	newBlock->mzs = (int *)mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	memcpy(newBlock->mzs, orig->mzs, dim * sizeof(int));

	// Copy all matrices (not just pointers)
	newBlock->ops = (MAT_TYPE **)mkl_malloc(newBlock->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i = 0; i < newBlock->num_ops; i++) {
		newBlock->ops[i] = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(newBlock->ops[i], orig->ops[i], dim*dim * sizeof(MAT_TYPE));
	}

	newBlock->psi = (MAT_TYPE *)mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(newBlock->psi, orig->psi, dim * sizeof(MAT_TYPE));
	newBlock->A = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(newBlock->A, orig->A, dim*dim * sizeof(MAT_TYPE));

	newBlock->d_trans = orig->d_trans;

	if (orig->trans == NULL) {
		newBlock->trans = NULL;
	} else {
		newBlock->trans = mkl_malloc(dim*newBlock->d_trans * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(newBlock->trans, orig->trans, dim*newBlock->d_trans * sizeof(MAT_TYPE));
	}

	newBlock->energy = orig->energy;
	newBlock->trunc_err = orig->trunc_err;

	return newBlock;
}

void freeDMRGBlock(DMRGBlock *block) {

	int i;
	for (i=0; i<block->num_ops; i++) {
		mkl_free(block->ops[i]);
	}
	mkl_free(block->ops);
	mkl_free(block->mzs);
	if (block->trans) { mkl_free(block->trans); }
	mkl_free(block->psi);
	mkl_free(block->A);

	mkl_free(block);
}

void printDMRGBlock(const char *desc, DMRGBlock *block) {
	printf("\n----------\n %s\n", desc);

	printf("length: %d\n", block->length);
	printf("d_block: %d\n", block->d_block);
	printf("num_ops: %d\n", block->num_ops);

	print_matrix("H", block->d_block, block->d_block, block->ops[0], block->d_block);
	print_matrix("conn_Sz", block->d_block, block->d_block, block->ops[1], block->d_block);
	print_matrix("conn_Sp", block->d_block, block->d_block, block->ops[2], block->d_block);

	printf("\n");
}

/* Print nice graphic of the system and environment
*/
void printGraphic(DMRGBlock *sys, DMRGBlock *env) {

	char *sys_g = (char *)malloc((sys->length +1) * sizeof(char));
	char *env_g = (char *)malloc((env->length +1) * sizeof(char));

	memset(sys_g, '=', sys->length);
	memset(env_g, '-', env->length);
	sys_g[sys->length] = '\0';
	env_g[env->length] = '\0';

	if (sys->side == 'L') {
		printf("%s**%s\n", sys_g, env_g);
	} else {
		printf("%s**%s\n", env_g, sys_g);
	}

	free(sys_g);
	free(env_g);
}

DMRGBlock *enlargeBlock(const DMRGBlock *block) {

	DMRGBlock *enl_block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
	enl_block->length  = block->length + 1;
	enl_block->fullLength = block->fullLength;
	int d_model = block->model->d_model;
	int dim = block->d_block * d_model;
	enl_block->d_block = dim;
	enl_block->num_ops = block->num_ops;
	enl_block->model   = block->model;
	enl_block->side    = block->side;
	enl_block->meas    = block->meas;

	enl_block->psi = (MAT_TYPE *)mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	enl_block->A = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	// TODO: add enlarged block A

	enl_block->ops = enlargeOps(block);
	if (enl_block->meas == 'M') { // measurement block
		enl_block->num_ops++;
	}

	enl_block->mzs = (int *)mkl_malloc(enl_block->d_block * sizeof(int), MEM_DATA_ALIGN);
	for (int i = 0; i < block->d_block; i++) {
		for (int j = 0; j < d_model; j++) {
			enl_block->mzs[i*d_model + j] = block->mzs[i] + block->model->init_mzs[j];
		}
	}

	enl_block->d_trans = 0;
	enl_block->trans = NULL;

	return enl_block;
}

/*  Operator Dictionary:
	0: H
	1: conn_Sz
	2: conn_Sp

	----------

	H_enl = kron(H, I_d) + kron(I_m, H1) + H_int(conn_Sz, conn_Sp, Sz, Sp)
	conn_Sz = kron(I_m, Sz)
	conn_Sp = kron(I_m, Sp)
*/

MAT_TYPE **enlargeOps(const DMRGBlock *block) {

	int numOps = block->num_ops;

	if (block->meas == 'M') {
		numOps++; // add new s_i block
	}

	MAT_TYPE **enl_ops = (MAT_TYPE **)mkl_malloc(numOps * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

	model_t *model = block->model;
	int d_model	= model->d_model;
	int d_block	= block->d_block;
	int d_enl  	= d_model * d_block;

	// H_enl
	enl_ops[0] = model->H_int(model->H_params, block, model->single_block);
	kronI('R', d_block, d_model, block->ops[0], enl_ops[0]);
	kronI('L', d_block, d_model, model->H1, enl_ops[0]);

	// conn_Sz
	enl_ops[1] = (MAT_TYPE *)mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sz, enl_ops[1]);

	// conn_Sp
	enl_ops[2] = (MAT_TYPE *)mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sp, enl_ops[2]);

	for (int i = 3; i < block->num_ops; i++) { // loop over measurement ops
		// S_i ops
		enl_ops[i] = (MAT_TYPE *)mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		kronI('R', d_block, d_model, block->ops[i], enl_ops[i]);

		// TODO: S_i S_j corrs for arbitary i and j
	}

	if (block->meas == 'M') {
		enl_ops[block->num_ops] = (MAT_TYPE *)mkl_malloc(d_enl*d_enl * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		// New S_i is same as conn_Sz
		memcpy(enl_ops[block->num_ops], enl_ops[1], d_enl*d_enl * sizeof(MAT_TYPE));
	}

	return enl_ops;
}

/* Prepares block to keep track of measurement ops
   Note: It is assumed that the block is length 1
*/
void startMeasBlock(DMRGBlock *block) {

	// assert(block->length == 1);
	block->meas = 'M';

	int dim = block->d_block;

	block->ops[block->num_ops] = (MAT_TYPE *)mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	// New S_i is same as conn_Sz
	memcpy(block->ops[block->num_ops], block->ops[1], dim*dim * sizeof(MAT_TYPE));

	block->num_ops++;
}

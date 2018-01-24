#include "block.h"
#include "model.h"
#include "linalg.h"
#include "util.h"
#include <mkl.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <input_parser.h>


DMRGBlock *createDMRGBlock(const model_t *model) {

	DMRGBlock *block = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

	block->length = 1;
	block->side = 'L';
	block->meas = 'N';

	int dim = model->d_model;
	block->d_block = dim;
	block->num_ops = model->num_ops;
	block->model   = model;

	block->mzs = mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	memcpy(block->mzs, model->init_mzs, dim * sizeof(int));

	// copy operators
	block->ops = mkl_malloc(block->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i = 0; i < block->num_ops; i++) {
		block->ops[i] = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(block->ops[i], model->init_ops[i], dim*dim * sizeof(MAT_TYPE));
	}

	block->psi = mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	block->A   = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	block->d_trans = 0;
	block->trans = NULL;

	// Set energy to ridiculous value
	block->energy = DBL_MAX;
	block->trunc_err = 0;

	return block;
}

DMRGBlock *copyDMRGBlock(const DMRGBlock *orig) {

	DMRGBlock *newBlock = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

	newBlock->length  = orig->length;
	newBlock->side  = orig->side;
	newBlock->meas  = orig->meas;
	int dim = orig->d_block;
	newBlock->d_block = dim;
	newBlock->num_ops = orig->num_ops;
	newBlock->model   = orig->model;

	newBlock->mzs = mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	memcpy(newBlock->mzs, orig->mzs, dim * sizeof(int));

	// Copy all matrices (not just pointers)
	newBlock->ops = mkl_malloc(newBlock->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i = 0; i < newBlock->num_ops; i++) {
		newBlock->ops[i] = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(newBlock->ops[i], orig->ops[i], dim*dim * sizeof(MAT_TYPE));
	}

	newBlock->psi = mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(newBlock->psi, orig->psi, dim * sizeof(MAT_TYPE));
	newBlock->A = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
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

	for (int i=0; i<block->num_ops; i++) {
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

	char *sys_g = malloc((sys->length +1) * sizeof(char));
	char *env_g = malloc((env->length +1) * sizeof(char));

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

	DMRGBlock *enl_block = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
	enl_block->length  = block->length + 1;
	int d_model = block->model->d_model;
	int dim = block->d_block * d_model;
	enl_block->d_block = dim;
	enl_block->num_ops = block->num_ops;
	enl_block->model   = block->model;
	enl_block->side    = block->side;
	enl_block->meas    = block->meas;

	enl_block->psi = mkl_malloc(dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	enl_block->A = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	// TODO: add enlarged block A

	enl_block->ops = enlargeOps(block);
	if (enl_block->meas == 'M') { // measurement block
		enl_block->num_ops++;
	}

	enl_block->mzs = mkl_malloc(enl_block->d_block * sizeof(int), MEM_DATA_ALIGN);
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

	MAT_TYPE **enl_ops = mkl_malloc(numOps * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

	const model_t *model = block->model;
	int d_model	= model->d_model;
	int d_block	= block->d_block;
	int d_enl  	= d_model * d_block;

	// H_enl
	enl_ops[0] = model->H_int(model, block, model->single_block);
	kronI('R', d_block, d_model, block->ops[0], enl_ops[0]);
	kronI('L', d_block, d_model, model->H1, enl_ops[0]);

	// conn_Sz
	enl_ops[1] = mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sz, enl_ops[1]);

	// conn_Sp
	enl_ops[2] = mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sp, enl_ops[2]);

	for (int i = 3; i < block->num_ops; i++) { // loop over measurement ops
		// S_i ops
		enl_ops[i] = mkl_calloc(d_enl*d_enl, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		kronI('R', d_block, d_model, block->ops[i], enl_ops[i]);

		// TODO: S_i S_j corrs for arbitary i and j
	}

	if (block->meas == 'M') {
		enl_ops[block->num_ops] = mkl_malloc(d_enl*d_enl * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
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

	block->ops[block->num_ops] = mkl_malloc(dim*dim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	// New S_i is same as conn_Sz
	memcpy(block->ops[block->num_ops], block->ops[1], dim*dim * sizeof(MAT_TYPE));

	block->num_ops++;
}

/* Returns roughly the memory footprint of a DMRGBlock with given d_block.
   This is meant as an estimate to now when it is necessary to save blocks to disk.
   It is assumed that trans is saved as long as PRIMME is used.
*/

MKL_INT64 estimateBlockMemFootprint(int d_block, int num_ops) {

	MKL_INT64 nbytes = 0;

	// nbytes += sizeof(DMRGBlock);

	// ops pointers
	nbytes += num_ops * sizeof(MAT_TYPE *);
	// ops and A
	nbytes += (num_ops + 1) * d_block*d_block * sizeof(MAT_TYPE);
	//psi
	nbytes += d_block * sizeof(MAT_TYPE);
	// mzs
	nbytes += d_block * sizeof(MAT_TYPE);
	
	// trans
	nbytes += d_block*2*d_block * sizeof(MAT_TYPE);
	
	return nbytes;
}

/* Returns the memory footprint of saving DMRGBlock block
*/

MKL_INT64 getSavedBlockMemFootprint(DMRGBlock *block) {

	MKL_INT64 nbytes = 0;

	// nbytes += sizeof(DMRGBlock);

	// ops pointers
	nbytes += block->num_ops * sizeof(MAT_TYPE *);
	// ops and A
	nbytes += (block->num_ops + 1) * block->d_block*block->d_block * sizeof(MAT_TYPE);
	//psi
	nbytes += block->d_block * sizeof(MAT_TYPE);
	// mzs
	nbytes += block->d_block * sizeof(MAT_TYPE);

	// trans
	if (block->trans != NULL) { nbytes += block->d_block*block->d_trans * sizeof(MAT_TYPE); }
	
	return nbytes;
}

int saveBlock(char *filename, DMRGBlock *block) {

	FILE *m_f = fopen(filename, "wb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int matsize = block->d_block*block->d_block;
	int count;

	count = fwrite(block, sizeof(DMRGBlock), 1, m_f);
	if (count != 1) {
		errprintf("Block not written properly to file '%s'.\n", filename);
		return -2;
	}

	for (int i=0; i<block->num_ops; i++) {
		count = fwrite(block->ops[i], sizeof(MAT_TYPE), matsize, m_f);
		if (count != matsize) {
			errprintf("Matrix '%d' not written properly to file '%s'.\n", i, filename);
			return -2;
		}
	}

	// matrix A
	count = fwrite(block->A, sizeof(MAT_TYPE), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix 'A' not written properly to file '%s'.\n", filename);
		return -2;
	}

	// psi
	count = fwrite(block->psi, sizeof(MAT_TYPE), block->d_block, m_f);
	if (count != block->d_block) {
		errprintf("Vector 'psi' not written properly to file '%s'.\n", filename);
		return -2;
	}

	// mzs
	count = fwrite(block->mzs, sizeof(MAT_TYPE), block->d_block, m_f);
	if (count != block->d_block) {
		errprintf("Vector 'mzs' not written properly to file '%s'.\n", filename);
		return -2;
	}

	// trans
	if (block->trans != NULL) {
		count = fwrite(block->trans, sizeof(MAT_TYPE), block->d_block*block->d_trans, m_f);
		if (count != block->d_block*block->d_trans) {
			errprintf("Matrix 'trans' not written properly to file '%s'.\n", filename);
			return -2;
		}
	}

	fclose(m_f);

	// if save succeeded, free all the saved memory
	for (int i=0; i<block->num_ops; i++) { mkl_free(block->ops[i]); }
	mkl_free(block->ops);
	mkl_free(block->A);
	mkl_free(block->psi);
	mkl_free(block->mzs);
	if (block->trans != NULL) { mkl_free(block->trans); }

	return 0;
}

int readBlock(char *filename, DMRGBlock *block) {

	FILE *m_f = fopen(filename, "rb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	// do not overwrite model which could change from run to run
	const model_t *model = block->model;

	int count;
	count = fread(block, sizeof(DMRGBlock), 1, m_f);
	if (count != 1) {
		errprintf("Block not written properly to file '%s'.\n", filename);
		return -2;
	}

	block->model = model;

	int matsize = block->d_block*block->d_block;
	block->ops = mkl_malloc(block->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i=0; i<block->num_ops; i++) {
		block->ops[i] = mkl_malloc(matsize * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		count = fread(block->ops[i], sizeof(MAT_TYPE), matsize, m_f);
		if (count != matsize) {
			errprintf("Matrix '%d' not read properly from file '%s'. Expected %d items but read %d.\n", i, filename, matsize, count);
			return -2;
		}
	}

	// matrix A
	block->A = mkl_malloc(matsize * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	count = fread(block->A, sizeof(MAT_TYPE), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix 'A' not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}

	// psi
	block->psi = mkl_malloc(block->d_block * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	count = fread(block->psi, sizeof(MAT_TYPE), block->d_block, m_f);
	if (count != block->d_block) {
		errprintf("Vector 'psi' not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}

	// mzs
	block->mzs = mkl_malloc(block->d_block * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	count = fread(block->mzs, sizeof(MAT_TYPE), block->d_block, m_f);
	if (count != block->d_block) {
		errprintf("Vector 'mzs' not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}

	// trans
	if (block->trans != NULL) {
		block->trans = mkl_malloc(block->d_block*block->d_trans * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		count = fread(block->trans, sizeof(MAT_TYPE), block->d_block*block->d_trans, m_f);
		if (count != block->d_block*block->d_trans) {
			errprintf("Matrix 'trans' not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
			return -2;
		}
	}

	fclose(m_f);

	return 0;
}
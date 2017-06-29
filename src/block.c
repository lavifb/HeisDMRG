#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>
#include <assert.h>


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

	int i;
	// copy operators
	block->ops = (double **)mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);
	for (i = 0; i < block->num_ops; i++) {
		block->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
		memcpy(block->ops[i], model->init_ops[i], dim*dim * sizeof(double));
	}

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

	int i;
	// Copy all matrices (not just pointers)
	newBlock->ops = (double **)mkl_malloc(newBlock->num_ops * sizeof(double *), MEM_DATA_ALIGN);
	for (i = 0; i < newBlock->num_ops; i++) {
		newBlock->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
		memcpy(newBlock->ops[i], orig->ops[i], dim*dim * sizeof(double));
	}

	return newBlock;
}

void freeDMRGBlock(DMRGBlock *block) {
	
	int i;
	for (i=0; i<block->num_ops; i++) {
		mkl_free(block->ops[i]);
	}
	mkl_free(block->ops);
	mkl_free(block->mzs);

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

DMRGBlock *enlargeBlock(const DMRGBlock *block) {

	DMRGBlock *enl_block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
	enl_block->length  = block->length + 1;
	enl_block->fullLength = block->fullLength;
	int d_model = block->model->d_model;
	enl_block->d_block = block->d_block * d_model;
	enl_block->num_ops = block->num_ops;
	enl_block->model   = block->model;
	enl_block->side    = block->side;
	enl_block->meas    = block->meas;

	enl_block->ops = enlargeOps(block);
	if (enl_block->meas == 'M') { // measurement block
		enl_block->num_ops++;
	}

	enl_block->mzs = (int *)mkl_malloc(enl_block->d_block * sizeof(int), MEM_DATA_ALIGN);
	int i, j;
	for (i = 0; i < block->d_block; i++) {
		for (j = 0; j < d_model; j++) {
			enl_block->mzs[i*d_model + j] = block->mzs[i] + block->model->init_mzs[j];
		}
	}

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

double **enlargeOps(const DMRGBlock *block) {

	int numOps = block->num_ops;

	if (block->meas == 'M') { 
		numOps++; // add new s_i block
	}

	double **enl_ops = (double **)mkl_malloc(numOps * sizeof(double *), MEM_DATA_ALIGN);

	model_t *model = block->model;
	int d_model	= model->d_model;
	int d_block	= block->d_block;
	int d_enl  	= d_model * d_block;

	double *I_m = identity(d_block);

	// H_enl
	enl_ops[0] = HeisenH_int(model->J, model->Jz, d_block, d_model, 
					block->ops[1], block->ops[2], model->Sz, model->Sp);
	kronI('R', d_block, d_model, block->ops[0], enl_ops[0]);
	kronI('L', d_block, d_model, model->H1, enl_ops[0]);

	// conn_Sz
	enl_ops[1] = (double *)mkl_calloc(d_enl*d_enl, sizeof(double), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sz, enl_ops[1]);

	// conn_Sp
	enl_ops[2] = (double *)mkl_calloc(d_enl*d_enl, sizeof(double), MEM_DATA_ALIGN);
	kronI('L', d_block, d_model, model->Sp, enl_ops[2]);

	int i;
	for (i = 3; i < block->num_ops; i++) { // loop over measurement ops
		// S_i ops
		enl_ops[i] = (double *)mkl_calloc(d_enl*d_enl, sizeof(double), MEM_DATA_ALIGN);
		kronI('R', d_block, d_model, block->ops[i], enl_ops[i]);

		// TODO: S_i S_j corrs for arbitary i and j
	}

	if (block->meas == 'M') {
		enl_ops[block->num_ops] = (double *)mkl_malloc(d_enl*d_enl * sizeof(double), MEM_DATA_ALIGN);
		// New S_i is same as conn_Sz
		memcpy(enl_ops[block->num_ops], enl_ops[1], d_enl*d_enl * sizeof(double));
	}

	mkl_free(I_m);
	return enl_ops;
}

/*  Transform an entire set of operators at once.
*/
void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops) {

	double *newOp = (double *)mkl_malloc(newDim*newDim * sizeof(double), MEM_DATA_ALIGN);
	double *temp  = (double *)mkl_malloc(newDim*opDim  * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);
	__assume_aligned(newOp, MEM_DATA_ALIGN);
	__assume_aligned(temp , MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < numOps; i++) {
		__assume_aligned(ops[i], MEM_DATA_ALIGN);
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, ops[i], opDim, 0.0, temp, newDim);
		cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, newOp, newDim);
		ops[i] = (double *)mkl_realloc(ops[i], newDim*newDim * sizeof(double));
		memcpy(ops[i], newOp, newDim*newDim * sizeof(double)); // copy newOp back into ops[i]
	}

	mkl_free(temp);
	mkl_free(newOp);
}
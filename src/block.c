#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>
#include <assert.h>


DMRGBlock *createDMRGBlock(ModelParams *model) {
	DMRGBlock *block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

	block->length = 1;
	block->side = 'L';

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
	newBlock->side  = orig->side;
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
	int d_model = block->model->d_model;
	enl_block->d_block = block->d_block * d_model;
	enl_block->num_ops = block->num_ops;
	enl_block->model   = block->model;
	enl_block->side    = block->side;

	enl_block->ops = enlargeOps(block);

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
	double **enl_ops = (double **)mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);

	ModelParams *model = block->model;
	int d_model	= model->d_model;
	int d_block	= block->d_block;
	int d_enl  	= d_model * d_block;

	double *I_m = identity(d_block);

	// H_enl
	enl_ops[0] = HeisenH_int(model->J, model->Jz, d_block, d_model, 
					block->ops[1], block->ops[2], model->Sz, model->Sp);
	kron(1.0, d_block, d_model, block->ops[0], model->Id, enl_ops[0]);
	kron(1.0, d_block, d_model, I_m, model->H1, enl_ops[0]);

	// conn_Sz
	enl_ops[1] = (double *)mkl_calloc(d_enl*d_enl, sizeof(double), MEM_DATA_ALIGN);
	kron(1.0, d_block, d_model, I_m, model->Sz, enl_ops[1]);

	// conn_Sp
	enl_ops[2] = (double *)mkl_calloc(d_enl*d_enl, sizeof(double), MEM_DATA_ALIGN);
	kron(1.0, d_block, d_model, I_m, model->Sp, enl_ops[2]);

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

/*	Create hashtable to track indexes of basis vectors corresponding to qns (called a sector)
*/
sector_t *sectorize(const DMRGBlock *block) {

	sector_t *secs = NULL; // Initialize uthash

	int j;
	for (j = 0; j < block->d_block; j++) {
		int id = block->mzs[j];
		sector_t *qn_sec;

		HASH_FIND_INT(secs, &id, qn_sec);

		if (qn_sec == NULL) { // create new entry in hashtable
			qn_sec = (sector_t *)mkl_malloc(sizeof(sector_t), MEM_DATA_ALIGN);
			qn_sec->id = id;
			qn_sec->num_ind = 1;
			qn_sec->inds[0] = j;
			HASH_ADD_INT(secs, id, qn_sec);
		} else { // add to entry
			assert(qn_sec->num_ind < HASH_IND_SIZE);
			qn_sec->inds[qn_sec->num_ind] = j;
			qn_sec->num_ind++;
		}
	}

	return secs;
}

/*	free sector hashtable
*/
void freeSector(sector_t *sectors) {

	sector_t *sec, *tmp;

	HASH_ITER(hh, sectors, sec, tmp) {
		HASH_DEL(sectors, sec);  // delete from hash
		mkl_free(sec);           // free sector
	}
}
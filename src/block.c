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
	block->num_qns = model->num_qns;
	block->model   = model;

	int i;
	// copy operators
	block->ops = (double **)mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);
	for (i = 0; i < block->num_ops; i++) {
		block->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
		memcpy(block->ops[i], model->init_ops[i], dim*dim * sizeof(double));
	}

	// copy qns
	block->qns = (int **)mkl_malloc(block->num_qns * sizeof(int *), MEM_DATA_ALIGN);
	for (i = 0; i < block->num_qns; i++) {
		block->qns[i] = (int *)mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
		memcpy(block->qns[i], model->init_qns[i], dim * sizeof(int));
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
	newBlock->num_qns    = orig->num_qns;
	newBlock->model   = orig->model;

	int i;
	// Copy all matrices (not just pointers)
	newBlock->ops = (double **)mkl_malloc(newBlock->num_ops * sizeof(double *), MEM_DATA_ALIGN);
	for (i = 0; i < newBlock->num_ops; i++) {
		newBlock->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
		memcpy(newBlock->ops[i], orig->ops[i], dim*dim * sizeof(double));
	}

	newBlock->qns = (int **)mkl_malloc(newBlock->num_qns * sizeof(int *), MEM_DATA_ALIGN);
	for (i = 0; i < newBlock->num_qns; i++) {
		newBlock->qns[i] = (int *)mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
		memcpy(newBlock->qns[i], orig->qns[i], dim * sizeof(int));
	}
	
	return newBlock;
}

void freeDMRGBlock(DMRGBlock *block) {
	
	int i;
	for (i=0; i<block->num_ops; i++) {
		mkl_free(block->ops[i]);
	}
	mkl_free(block->ops);

	for (i=0; i<block->num_qns; i++) {
		mkl_free(block->qns[i]);
	}
	mkl_free(block->qns);

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
	enl_block->d_block  = block->d_block * block->model->d_model;
	enl_block->num_ops = block->num_ops;
	enl_block->model   = block->model;
	enl_block->side    = block->side;
	enl_block->num_qns = block->num_qns;

	enl_block->ops = enlargeOps(block);

	enl_block->qns = enlargeQns(block);

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

/* Returns qns for enlarged block
*/
int **enlargeQns(const DMRGBlock *block) {
	int **enl_qns = (int **)mkl_malloc(block->num_qns * sizeof(int *), MEM_DATA_ALIGN);

	int d_model	= block->model->d_model;
	int d_block	= block->d_block;
	int d_enl = d_block * d_model;

	int i, j, k;
	for (i = 0; i < block->num_qns; i++) {
		enl_qns[i] = (int *)mkl_malloc(d_enl * sizeof(int), MEM_DATA_ALIGN);
		for (j = 0; j < d_block; j++) {
			for (k = 0; k < d_model; k++) {
				// TODO: Make this more general. In general qns don't just add. 
				enl_qns[i][j*d_model + k] = block->qns[i][j] + block->model->init_qns[i][k];
			}
		}
	}
	
	return enl_qns;
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
sector_t **sectorize(const DMRGBlock *block) {

	sector_t **secs = (sector_t **)mkl_malloc(block->num_qns * sizeof(sector_t), MEM_DATA_ALIGN);

	int i, j;
	for (i = 0; i < block->num_qns; i++) {
		secs[i] = NULL; // Initialize uthash
	}

	for (j = 0; j < block->d_block; j++) {
		for (i = 0; i < block->num_qns; i++) {
			sector_t *qn_sec;
			int id = block->qns[i][j];

			HASH_FIND_INT(secs[i], &id, qn_sec);

			if (qn_sec == NULL) { // create new entry in hashtable
				qn_sec = (sector_t *)mkl_malloc(sizeof(sector_t), MEM_DATA_ALIGN);
				qn_sec->id = id;
				qn_sec->num_ind = 1;
				qn_sec->inds[0] = j;
				HASH_ADD_INT(secs[i], id, qn_sec);
			} else { // add to entry
				assert(qn_sec->num_ind < HASH_IND_SIZE);
				qn_sec->inds[qn_sec->num_ind] = j;
				qn_sec->num_ind++;
			}
		}
	}

	return secs;
}

void freeSectors(sector_t **secs) {

}
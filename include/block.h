#ifndef BLOCK_H
#define BLOCK_H

#include "linalg.h"

// farward declaration for model
typedef struct model_t model_t;

typedef struct DMRGBlock {
	int length;
	int fullLength;
	char side;  	// 'L' if left block and 'R' if right block
	char meas;  	// 'N' is normal block and 'M' if measurements are tracked
	int d_block;	// dimension of basis
	int num_ops;
	MAT_TYPE **ops;
	int *mzs;   	// 2*mz quantum number for each state
	model_t *model;

	int d_trans;    // dimension of basis before transfoming
	MAT_TYPE *trans;// transformation matrix used to truncate the block

	MAT_TYPE *psi;	// tracked state
	MAT_TYPE *A;  	// operator that takes ground to desired state: A|psi0> = |psi>

	double energy;
	double trunc_err;
} DMRGBlock;

DMRGBlock *createDMRGBlock(model_t *model, int fullLength);

DMRGBlock *copyDMRGBlock(DMRGBlock *orig);

void freeDMRGBlock(DMRGBlock *block);

void printDMRGBlock(const char *desc, DMRGBlock *block);

void printGraphic(DMRGBlock *sys, DMRGBlock *env);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

MAT_TYPE **enlargeOps(const DMRGBlock *block);

void startMeasBlock(DMRGBlock *block);

MKL_INT64 estimateBlockMemFootprint(int d_block, int num_ops);

MKL_INT64 getSavedBlockMemFootprint(DMRGBlock *block);

int saveBlock(char *filename, DMRGBlock *block);

int readBlock(char *filename, DMRGBlock *block);

#endif
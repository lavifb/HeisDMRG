#ifndef BLOCK_H
#define BLOCK_H

#include "uthash.h"
#include "model.h"

typedef struct {
	int length;
	int fullLength;
	char side;
	int d_block;	// dimension of basis
	double **ops;
	int num_ops;
	int *mzs;		// 2*mz quantum number for each state
	ModelParams *model;
} DMRGBlock;

DMRGBlock *createDMRGBlock(ModelParams *model);

DMRGBlock *copyDMRGBlock(DMRGBlock *orig);

void freeDMRGBlock(DMRGBlock *block);

void printDMRGBlock(const char *desc, DMRGBlock *block);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

double **enlargeOps(const DMRGBlock *block);

void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops);

#endif
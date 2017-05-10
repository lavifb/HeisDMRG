#ifndef BLOCK_H
#define BLOCK_H

// #include "uthash.h"
#include "model.h"

typedef struct {
	int length;
	char side;
	int d_block; // dimension of basis
	double **ops;
	int num_ops;
	ModelParams *model;

	int num_qns;	// number of quantum numbers
	int **qns;  	// list of quantum numbers for single site 
	            	// each initqns[i] has size d_model)
} DMRGBlock;

DMRGBlock *createDMRGBlock(ModelParams *model);

DMRGBlock *copyDMRGBlock(DMRGBlock *orig);

void freeDMRGBlock(DMRGBlock *block);

void freeDMRGBlockOps(DMRGBlock *block);

void printDMRGBlock(const char *desc, DMRGBlock *block);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

double **enlargeOps(const DMRGBlock *block);

int **enlargeQns(const DMRGBlock *block);

void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops);

#endif
#ifndef BLOCK_H
#define BLOCK_H

#include "uthash.h"
#include "model.h"

typedef struct {
	int length;
	int fullLength;
	char side;  	// 'L' if left block and 'R' if right block
	char meas;  	// 'N' is normal block and 'M' if measurements are tracked
	int d_block;	// dimension of basis
	int num_ops;
	double **ops;
	int *mzs;   	// 2*mz quantum number for each state
	model_t *model;

	double *psi;	// tracked state
	double *A;  	// operator that takes ground to desired state: A|psi0> = |psi>

	double energy;
	double trunc_err;
} DMRGBlock;

DMRGBlock *createDMRGBlock(model_t *model, int fullLength);

DMRGBlock *copyDMRGBlock(DMRGBlock *orig);

void freeDMRGBlock(DMRGBlock *block);

void printDMRGBlock(const char *desc, DMRGBlock *block);

void printGraphic(DMRGBlock *sys, DMRGBlock *env);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

double **enlargeOps(const DMRGBlock *block);

void startMeasBlock(DMRGBlock *block);

#endif
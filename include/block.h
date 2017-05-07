#ifndef BLOCK_H
#define BLOCK_H

// #include "uthash.h"
#include "model.h"

typedef struct {
    int length;
    char side;
    int dBlock; // dimension of basis
    double **ops;
    int num_ops;
    ModelParams *model;
} DMRGBlock;

DMRGBlock *createDMRGBlock(ModelParams *model);

void freeDMRGBlock(DMRGBlock *block);

void freeDMRGBlockOps(DMRGBlock *block);

void printDMRGBlock(const char *desc, DMRGBlock *block);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

double **enlargeOps(const DMRGBlock *block);

void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops);

#endif
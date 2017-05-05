#ifndef BLOCK_H
#define BLOCK_H

// #include "uthash.h"
#include "model.h"

typedef struct {
    int length;
    int basis_size;
    // TODO: Maybe restrict?
    double **ops;
    int num_ops;
    ModelParams *model;
} DMRGBlock;

void freeDMRGBlock(DMRGBlock *block);

DMRGBlock *enlargeBlock(const DMRGBlock *block);

double **enlargeOps(const DMRGBlock *block);

void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops);

#endif
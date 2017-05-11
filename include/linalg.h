#ifndef LINALG_H
#define LINALG_H

void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C);

double *identity(const int N);

double *transformOp(const int opDim, const int newDim, const double *restrict trans, const double *restrict op);

double *restrictOp(const int m, const double *op, const int num_ind, const int *inds);

double *restrictVec(const int m, double *v, const int num_ind, const int *inds);

void print_matrix( char* desc, int m, int n, double* a, int lda );

#endif
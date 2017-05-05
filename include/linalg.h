#ifndef LINALG_H
#define LINALG_H

void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C);

double *identity(const int N);

double *transformOp(const int opDim, const int newDim, const double *restrict trans, const double *restrict op);

#endif
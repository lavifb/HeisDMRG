#ifndef LINALG_H
#define LINALG_H

void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C);

double *identity(const int N);

#endif LINALG_H
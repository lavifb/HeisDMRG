#ifndef LINALG_H
#define LINALG_H

#include <mkl_types.h>

#if COMPLEX
#define MAT_TYPE MKL_Complex16
#else
#define MAT_TYPE double
#endif


void kron(const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B, MAT_TYPE *restrict C);

void kron_r(const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B,
	        MAT_TYPE *restrict C, const int num_ind, const int *restrict inds);

void kronT(const char side, const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B, MAT_TYPE *restrict C);

void kronT_r(const char side, const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B,
			MAT_TYPE *restrict C, const int num_ind, const int *restrict inds);

void kronI(const char side, const int m, const int n, const MAT_TYPE *restrict A, MAT_TYPE *restrict C);

void kronI_r(const char side, const int m, const int n, const MAT_TYPE *restrict A, 
	         MAT_TYPE *restrict C, const int num_ind, const int *restrict inds);

MAT_TYPE *identity(const int N);

MAT_TYPE *transformOp(const int opDim, const int newDim, const MAT_TYPE *restrict trans, const MAT_TYPE *restrict op);

void transformOps(const int numOps, const int opDim, const int newDim, const MAT_TYPE *restrict trans, MAT_TYPE **ops);

MAT_TYPE *restrictOp(const int m, const MAT_TYPE *op, const int num_ind, const int *inds);

MAT_TYPE *restrictVec(const MAT_TYPE *v, const int num_ind, const int *inds);

MAT_TYPE *unrestrictVec(const int m, const MAT_TYPE *v_r, const int num_ind, const int *inds);

int *dsort2(const int n, double *a);

void print_matrix(char* desc, int m, int n, MAT_TYPE *a, int lda);

#endif
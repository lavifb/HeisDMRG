#include "linalg.h"
#include <mkl.h>
#include <stdlib.h>

/*
	Compute Kronecker product of two square matrices. Sets C = alpha * kron(A,B) + C
 
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix
	C: mn*mn matrix
*/
void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C) {
	int ldac = m*n;
	int i, j, k, l;
	for (i=0; i<m; i++) {
		for (j=0; j<m; j++) {
			for (k=0; k<n; k++) {
				for (l=0; l<n; l++) {
					C[(n*i + k) + ldac*(n*j + l)] += alpha * A[i+m*j]*B[k+n*l];
				}
			}
		}
	}
}

/* Creates identity matrix of size N*N
*/
double *identity(const int N) {
	double *Id = (double *)mkl_calloc(N*N, sizeof(double), MEM_DATA_ALIGN);
	int i;
	for (i = 0; i < N; i++) Id[i*N+i] = 1.0;

	return Id;
}

/* 
	Transforms matrix into new truncated basis. returns trans^T * op * trans
*/
double *transformOp(const int opDim, const int newDim, const double *restrict trans, const double *restrict op) {

	double *newOp = (double *)mkl_malloc(newDim*newDim * sizeof(double), MEM_DATA_ALIGN);
	double *temp  = (double *)mkl_malloc(newDim*opDim  * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(op   , MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);
	__assume_aligned(newOp, MEM_DATA_ALIGN);
	__assume_aligned(temp , MEM_DATA_ALIGN);

	cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, op, opDim, 0.0, temp, newDim);
	cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, newOp, newDim);

	mkl_free(temp);
	return newOp;
}

/*  Restrict square matrix to only indexes 

	m       : dimension of op
	op      : m*m matrix
	num_ind : number of indexes provided. Also dimension of output
	inds    : list of indexes
*/
double *restrictOp(const int m, const double *op, const int num_ind, const int *inds) {

	double *op_r = (double *)mkl_malloc(num_ind*num_ind * sizeof(double), MEM_DATA_ALIGN);

	int i, j;
	for (i = 0; i < num_ind; i++) {
		for (j = 0; j < num_ind; j++) {
			int op_i = inds[i]*m + inds[j];
			op_r[i*num_ind + j] = op[op_i];
		}
	}

	return op_r;
}

/*  Restrict vector to only indexes 

	m       : dimension of v
	v       : input vector
	num_ind : number of indexes provided. Also dimension of output
	inds    : list of indexes
*/
double *restrictVec(const int m, const double *v, const int num_ind, const int *inds) {

	double *v_r = (double *)mkl_malloc(num_ind * sizeof(double), MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < num_ind; i++) {
		v_r[i] = v[inds[i]];
	}

	return v_r;
}

/*  Unrestrict vector 

	m       : dimension of v
	v_r     : input restricted vector
	num_ind : number of indexes provided. Also dimension of output
	inds    : list of indexes
*/
double *unrestrictVec(const int m, const double *v_r, const int num_ind, const int *inds) {

	double *v = (double *)mkl_calloc(m, sizeof(double), MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < num_ind; i++) {
		v[inds[i]] = v_r[i];
	}

	return v;
}

// Pointer comparison for sort below
int dcmp(const void *pa, const void *pb){
    double a = **(double **)pa;
    double b = **(double **)pb;

    return b - a > 0 ? 1 : -1;
}

/* Sort in descending order. Returns indexes in sorted order
*/
int *dsort2(const int n, double *a) {

	double **ais = (double **)mkl_malloc(n * sizeof(double *), MEM_DATA_ALIGN);
	double *temp = (double *)mkl_malloc(n * sizeof(double), MEM_DATA_ALIGN);
	memcpy(temp, a, n * sizeof(double));

	int i;
	for(i = 0; i < n; i++) {
		ais[i] = &temp[i];
	}

	double *ai0 = ais[0];
	qsort(ais, n, sizeof(ais[0]), dcmp);
	
	int *inds = (int *)mkl_malloc(n * sizeof(int), MEM_DATA_ALIGN);
	for(i = 0; i < n; i++) {
		inds[i] = (int) (ais[i] - ai0);
		a[i] = *ais[i];
	}

	mkl_free(temp);
	mkl_free(ais);
	return inds;
}

/* Print matrix from Intel MKL examples
*/
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
	int i, j;
	printf( "\n %s\n", desc );
	for(i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " % 6.2f", a[i+j*lda] );
		printf( "\n" );
	}
}
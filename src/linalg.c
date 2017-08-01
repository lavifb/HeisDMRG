#include "linalg.h"
#include <mkl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define ZERO_TOLERANCE 10e-7

/*  Compute Kronecker product of two square matrices. Sets C = alpha * kron(A,B) + C
 
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix
	C: mn*mn matrix
*/
void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	int i, j, k, l;
	for (i=0; i<m; i++) {
		for (j=0; j<m; j++) {
			if (A[i+m*j] == 0.0) { continue; }

			for (k=0; k<n; k++) {
				for (l=0; l<n; l++) {
					C[(n*i + k) + ldac*(n*j + l)] += B[k+n*l]*A[i+m*j] * alpha;
				}
			}
		}
	}
}

/*  Compute Kronecker product of two square matrices and restrict basis.
	Sets C = alpha * kron(A,B) + C within restricted basis.
 
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix

	num_ind: number of restricted basis inds and size of matrix C 
	inds:    restricted basis inds
	C: num_ind*num_ind matrix
*/
void kron_r(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B,
	        double *restrict C, const int num_ind, const int *restrict inds) {

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	int p, q;
	for (p=0; p<num_ind; p++) {
		for (q=0; q<num_ind; q++) {
			int i = inds[q]/n;
			int j = inds[p]/n;
			int k = inds[q]%n;
			int l = inds[p]%n;

			C[num_ind*p + q] += B[k+n*l]*A[i+m*j] * alpha;
		}
	}
}

/*  Compute Kronecker product of a square matrix with identity.
	Sets C = kron(A, I) + C or C = kron(I, A) + C

	Note: The order of m and n is designed to easily replace standard kron function
	without swapping parameters.

	side: 'l' or 'L' if I is multplied on the left or
	      'r' or 'R' if I is multplied on the right
	m: size of matrix A if side is 'r' or
	   size of identity matrix if side is 'l'
	n: size of identity matrix if side is 'r' or
	   size of matrix A if side is 'l'
	A: m*m matrix or n*n matrix
	C: mn*mn matrix
*/
void kronI(const char side, const int m, const int n, const double *restrict A, double *restrict C) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	int i, j, k;
	switch (side) {

		case 'r':
		case 'R':
			for (i=0; i<m; i++) {
				for (j=0; j<m; j++) {
					if (A[i+m*j] == 0.0) { continue; }

					for (k=0; k<n; k++) {
						C[(n*i + k) + ldac*(n*j + k)] += A[i+m*j];
					}
				}
			}
			break;

		case 'l':
		case 'L':
			for (i=0; i<n; i++) {
				for (j=0; j<n; j++) {
					if (A[i+n*j] == 0.0) { continue; }

					for (k=0; k<m; k++) {
						C[(n*k + i) + ldac*(n*k + j)] += A[i+n*j];
					}
				}
			}
			break;
	}
}

/*  Compute Kronecker product of a square matrix with identity and restrict basis.
	Sets C = kron(A, I) + C or C = kron(I, A) + C

	Note: The order of m and n is designed to easily replace standard kron function
	without swapping parameters.

	side: 'l' or 'L' if I is multplied on the left or
	      'r' or 'R' if I is multplied on the right
	m: size of matrix A if side is 'r' or
	   size of identity matrix if side is 'l'
	n: size of identity matrix if side is 'r' or
	   size of matrix A if side is 'l'
	A: m*m matrix or n*n matrix
	C: mn*mn matrix
*/
void kronI_r(const char side, const int m, const int n, const double *restrict A, 
	         double *restrict C, const int num_ind, const int *restrict inds) {

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	int p, q;
	switch (side) {

		default:
		case 'r':
		case 'R':
			for (p=0; p<num_ind; p++) {
				for (q=0; q<num_ind; q++) {
					int i = inds[q]/n;
					int j = inds[p]/n;

					if (inds[p]%n == inds[q]%n) {
						C[num_ind*p + q] += A[i+m*j];
					}
				}
			}
			break;

		case 'l':
		case 'L':
			for (p=0; p<num_ind; p++) {
				for (q=0; q<num_ind; q++) {
					int i = inds[q]%n;
					int j = inds[p]%n;

					if (inds[p]/n == inds[q]/n) {
						C[num_ind*p + q] += A[i+n*j];
					}
				}
			}
			break;
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

/*  Transforms matrix into new truncated basis. returns trans^T * op * trans
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

/*  Transform an entire set of operators at once.
	Note that ops is changed instead of returning the transformed ops.
*/
void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops) {

	double *newOp = (double *)mkl_malloc(newDim*newDim * sizeof(double), MEM_DATA_ALIGN);
	double *temp  = (double *)mkl_malloc(newDim*opDim  * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);
	__assume_aligned(newOp, MEM_DATA_ALIGN);
	__assume_aligned(temp , MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < numOps; i++) {
		__assume_aligned(ops[i], MEM_DATA_ALIGN);
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, ops[i], opDim, 0.0, temp, newDim);
		cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, newOp, newDim);
		ops[i] = (double *)mkl_realloc(ops[i], newDim*newDim * sizeof(double));
		memcpy(ops[i], newOp, newDim*newDim * sizeof(double)); // copy newOp back into ops[i]
	}

	mkl_free(temp);
	mkl_free(newOp);
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

	v       : input vector
	num_ind : number of indexes provided. Also dimension of output
	inds    : list of indexes
*/
double *restrictVec(const double *v, const int num_ind, const int *inds) {

	double *v_r = (double *)mkl_malloc(num_ind * sizeof(double), MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < num_ind; i++) {
		v_r[i] = v[inds[i]];
	}

	return v_r;
}
/*  Restrict vector to only indexes where v is nonzero.

	m       : dimension of v
	v       : input vector

	RETURNS
	v       : restricted input vector
	num_indp: pointer to number of unrestricted indexes
	inds    : list of indexes
*/
int *restrictVecToNonzero(const int m, double *v, int *num_indp) {

	int num_ind = 0;
	double *v_r = (double *)mkl_malloc(m * sizeof(double), MEM_DATA_ALIGN);
	int *inds = (int *)mkl_malloc(m * sizeof(int), MEM_DATA_ALIGN);

	int i;
	for (i = 0; i < m; i++) {
		if (fabs(v[i]) > ZERO_TOLERANCE) {
			v_r[num_ind] = v[i];
			inds[num_ind] = i;
			num_ind++;
		}
	}

	v = mkl_realloc(v, num_ind * sizeof(double));
	inds = mkl_realloc(inds, num_ind * sizeof(int));

	memcpy(v, v_r, num_ind * sizeof(double));
	mkl_free(v_r);

	*num_indp = num_ind;

	return inds;
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
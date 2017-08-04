#include "linalg.h"
#include <mkl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define ZERO_TOLERANCE 10e-7

/*  Sets y = alpha * x1 * x2
*/
inline void mult(const double alpha, const MAT_TYPE x1, const MAT_TYPE x2, MAT_TYPE *y) {
	#if COMPLEX
	*y.real += alpha * (x1.real*x2.real - x1.imag*x2.imag);
	*y.imag += alpha * (x1.real*x2.imag + x1.imag*x2.real);
	#else
	*y += alpha * x1 * x2;
	#endif
}

#if COMPLEX
	#define MULT(alpha, x1, x2, y) y.real += alpha * (x1.real*x2.real - x1.imag*x2.imag); y.imag += alpha * (x1.real*x2.imag + x1.imag*x2.real);
#else
	#define MULT(alpha, x1, x2, y) y += alpha * x1 * x2;
#endif

#if COMPLEX
	#define PLUSEQ(x, y) y.real += x.real; y.imag += x.imag;
#else
	#define PLUSEQ(x, y) y += x;
#endif

/*  Sets y += x
*/
inline void pluseq(const MAT_TYPE x, MAT_TYPE *y) {
	#if COMPLEX
	*y.real += x.real;
	*y.imag += x.imag;
	#else
	*y += x;
	#endif
}


/*  Compute Kronecker product of two square matrices. Sets C = alpha * kron(A,B) + C
 
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix
	C: mn*mn matrix
*/
void kron(const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B, MAT_TYPE *restrict C) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			#if COMPLEX
			if (A[i+m*j].real == 0.0 && A[i+m*j].imag == 0.0) { continue; }
			#else
			if (A[i+m*j] == 0.0) { continue; }
			#endif

			for (int k=0; k<n; k++) {
				for (int l=0; l<n; l++) {
					MULT(alpha, B[k+n*l], A[i+m*j], C[(n*i + k) + ldac*(n*j + l)]);
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
void kron_r(const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B,
	        MAT_TYPE *restrict C, const int num_ind, const int *restrict inds) {

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	for (int p=0; p<num_ind; p++) {
		for (int q=0; q<num_ind; q++) {
			int i = inds[q]/n;
			int j = inds[p]/n;
			int k = inds[q]%n;
			int l = inds[p]%n;

			MULT(alpha, B[k+n*l], A[i+m*j], C[num_ind*p + q]);
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
void kronI(const char side, const int m, const int n, const MAT_TYPE *restrict A, MAT_TYPE *restrict C) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	switch (side) {

		case 'r':
		case 'R':
			for (int i=0; i<m; i++) {
				for (int j=0; j<m; j++) {
					#if COMPLEX
					if (A[i+m*j].real == 0.0 && A[i+m*j].imag == 0.0) { continue; }
					#else
					if (A[i+m*j] == 0.0) { continue; }
					#endif

					for (int k=0; k<n; k++) {
						PLUSEQ(A[i+m*j], C[(n*i + k) + ldac*(n*j + k)]);
					}
				}
			}
			break;

		case 'l':
		case 'L':
			for (int i=0; i<n; i++) {
				for (int j=0; j<n; j++) {
					#if COMPLEX
					if (A[i+n*j].real == 0.0 && A[i+n*j].imag == 0.0) { continue; }
					#else
					if (A[i+n*j] == 0.0) { continue; }
					#endif

					for (int k=0; k<m; k++) {
						PLUSEQ(A[i+n*j], C[(n*k + i) + ldac*(n*k + j)]);
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
void kronI_r(const char side, const int m, const int n, const MAT_TYPE *restrict A, 
	         MAT_TYPE *restrict C, const int num_ind, const int *restrict inds) {

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	switch (side) {

		default:
		case 'r':
		case 'R':
			for (int p=0; p<num_ind; p++) {
				for (int q=0; q<num_ind; q++) {
					int i = inds[q]/n;
					int j = inds[p]/n;

					if (inds[p]%n == inds[q]%n) {
						PLUSEQ(A[i+m*j], C[num_ind*p + q]);
					}
				}
			}
			break;

		case 'l':
		case 'L':
			for (int p=0; p<num_ind; p++) {
				for (int q=0; q<num_ind; q++) {
					int i = inds[q]%n;
					int j = inds[p]%n;

					if (inds[p]/n == inds[q]/n) {
						PLUSEQ(A[i+n*j], C[num_ind*p + q]);
					}
				}
			}
			break;
	}
}

/* Creates identity matrix of size N*N
*/
MAT_TYPE *identity(const int N) {
	MAT_TYPE *Id = (MAT_TYPE *)mkl_calloc(N*N, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	for (int i = 0; i < N; i++) {
		#if COMPLEX
		Id[i*N+i].real = 1.0;
		#else
		Id[i*N+i] = 1.0;
		#endif
	}

	return Id;
}

/*  Transforms matrix into new truncated basis. returns trans^T * op * trans
*/
MAT_TYPE *transformOp(const int opDim, const int newDim, const MAT_TYPE *restrict trans, const MAT_TYPE *restrict op) {

	MAT_TYPE *newOp = (MAT_TYPE *)mkl_malloc(newDim*newDim * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *temp  = (MAT_TYPE *)mkl_malloc(newDim*opDim  * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	__assume_aligned(op   , MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);
	__assume_aligned(newOp, MEM_DATA_ALIGN);
	__assume_aligned(temp , MEM_DATA_ALIGN);

	#if COMPLEX
	MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, &one, trans, opDim, op, opDim, &zero, temp, newDim);
	cblas_zgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, &one, temp, newDim, trans, opDim, &zero, newOp, newDim);
	#else
	cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, op, opDim, 0.0, temp, newDim);
	cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, newOp, newDim);
	#endif

	mkl_free(temp);
	return newOp;
}

/*  Transform an entire set of operators at once.
	Note that ops is changed instead of returning the transformed ops.
*/
void transformOps(const int numOps, const int opDim, const int newDim, const MAT_TYPE *restrict trans, MAT_TYPE **ops) {

	MAT_TYPE *temp  = (MAT_TYPE *)mkl_malloc(newDim*opDim  * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);
	__assume_aligned(temp , MEM_DATA_ALIGN);

	for (int i = 0; i < numOps; i++) {
		__assume_aligned(ops[i], MEM_DATA_ALIGN);
		#if COMPLEX
		MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		MKL_Complex16 zero = {.real=0.0, .imag=0.0};
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, &one, trans, opDim, ops[i], opDim, &zero, temp, newDim);
		ops[i] = (MAT_TYPE *)mkl_realloc(ops[i], newDim*newDim * sizeof(MAT_TYPE));
		cblas_zgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, &one, temp, newDim, trans, opDim, &zero, ops[i], newDim);
		#else
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, ops[i], opDim, 0.0, temp, newDim);
		ops[i] = (MAT_TYPE *)mkl_realloc(ops[i], newDim*newDim * sizeof(MAT_TYPE));
		cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, ops[i], newDim);
		#endif
	}

	mkl_free(temp);
}

/*  Restrict square matrix to only indexes 

	m       : dimension of op
	op      : m*m matrix
	num_ind : number of indexes provided. Also dimension of output
	inds    : list of indexes
*/
MAT_TYPE *restrictOp(const int m, const MAT_TYPE *op, const int num_ind, const int *inds) {

	MAT_TYPE *op_r = (MAT_TYPE *)mkl_malloc(num_ind*num_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	for (int i = 0; i < num_ind; i++) {
		for (int j = 0; j < num_ind; j++) {
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
MAT_TYPE *restrictVec(const MAT_TYPE *v, const int num_ind, const int *inds) {

	MAT_TYPE *v_r = (MAT_TYPE *)mkl_malloc(num_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	for (int i = 0; i < num_ind; i++) {
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
int *restrictVecToNonzero(const int m, MAT_TYPE *v, int *num_indp) {

	int num_ind = 0;
	MAT_TYPE *v_r = (MAT_TYPE *)mkl_malloc(m * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	int *inds = (int *)mkl_malloc(m * sizeof(int), MEM_DATA_ALIGN);

	for (int i = 0; i < m; i++) {
		#if COMPLEX
		if (fabs(v[i].real) > ZERO_TOLERANCE || fabs(v[i].imag) > ZERO_TOLERANCE) {
		#else
		if (fabs(v[i]) > ZERO_TOLERANCE) {
		#endif
			v_r[num_ind] = v[i];
			inds[num_ind] = i;
			num_ind++;
		}
	}

	v = mkl_realloc(v, num_ind * sizeof(MAT_TYPE));
	inds = mkl_realloc(inds, num_ind * sizeof(int));

	memcpy(v, v_r, num_ind * sizeof(MAT_TYPE));
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
MAT_TYPE *unrestrictVec(const int m, const MAT_TYPE *v_r, const int num_ind, const int *inds) {

	MAT_TYPE *v = (MAT_TYPE *)mkl_calloc(m, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	for (int i = 0; i < num_ind; i++) {
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

	for (int i = 0; i < n; i++) {
		ais[i] = &temp[i];
	}

	double *ai0 = ais[0];
	qsort(ais, n, sizeof(ais[0]), dcmp);
	
	int *inds = (int *)mkl_malloc(n * sizeof(int), MEM_DATA_ALIGN);
	for (int i = 0; i < n; i++) {
		inds[i] = (int) (ais[i] - ai0);
		a[i] = *ais[i];
	}

	mkl_free(temp);
	mkl_free(ais);
	return inds;
}

/* Print matrix from Intel MKL examples
*/
void print_matrix(char* desc, int m, int n, MAT_TYPE *a, int lda) {
	printf("\n %s\n", desc);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			#if COMPLEX
			printf(" % 6.2f %+6.2fI", a[i+j*lda].real, a[i+j*lda].imag);
			#else
			printf(" % 6.2f", a[i+j*lda]);
			#endif
		}
		printf("\n");
	}
}
#include "linalg.h"
#include <mkl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>

#if USE_PRIMME
#include "primme.h"
#endif

#define ZERO_TOLERANCE 1e-9

#if COMPLEX
	#define PLUSEQ(x, y) y.real += x.real; y.imag += x.imag;
	#define MULT(alpha, x1, x2, y) y.real += alpha * (x1.real*x2.real - x1.imag*x2.imag); y.imag += alpha * (x1.real*x2.imag + x1.imag*x2.real);
	#define MULTCONJ(alpha, x1, x2, y) y.real += alpha * (x1.real*x2.real + x1.imag*x2.imag); y.imag += alpha * (x1.imag*x2.real - x1.real*x2.imag);
#else
	#define PLUSEQ(x, y) y += x;
	#define MULT(alpha, x1, x2, y) y += alpha * x1 * x2;
	#define MULTCONJ(alpha, x1, x2, y) y += alpha * x1 * x2;
#endif

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

	#pragma omp parallel for
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

/*  Compute Kronecker product of two square matrices with one conjugate transposed. 
	Sets C = alpha * kron(At,B) + C or C = alpha * kron(A,Bt) + C
 
 	side: 'l' or 'L' if left matrix transposed
 	      'r' or 'R' if right matrix transposed
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix
	C: mn*mn matrix
*/
void kronT(const char side, const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B, MAT_TYPE *restrict C) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
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
						for (int l=0; l<n; l++) {
							MULTCONJ(alpha, A[i+m*j], B[l+n*k], C[(n*i + k) + ldac*(n*j + l)]);
						}
					}
				}
			}
			break;
		case 'l':
		case 'L':
			for (int i=0; i<m; i++) {
				for (int j=0; j<m; j++) {
					#if COMPLEX
					if (A[j+m*i].real == 0.0 && A[j+m*i].imag == 0.0) { continue; }
					#else
					if (A[j+m*i] == 0.0) { continue; }
					#endif

					for (int k=0; k<n; k++) {
						for (int l=0; l<n; l++) {
							MULTCONJ(alpha, B[k+n*l], A[j+m*i], C[(n*i + k) + ldac*(n*j + l)]);
						}
					}
				}
			}
			break;
	}

}

/*  Compute Kronecker product of two square matrices with one conjugate transposed and restrict basis. 
	Sets C = alpha * kron(At,B) + C or C = alpha * kron(A,Bt) + C
 
 	side: 'l' or 'L' if left matrix transposed
 	      'r' or 'R' if right matrix transposed
	m: size of matrix A
	n: size of matrix B
	A: m*m matrix
	B: n*n matrix
	C: num_ind*num_ind matrix
*/
void kronT_r(const char side, const double alpha, const int m, const int n, const MAT_TYPE *restrict A, const MAT_TYPE *restrict B,
			MAT_TYPE *restrict C, const int num_ind, const int *restrict inds) {
	int ldac = m*n;

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(B, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	switch (side) {

		case 'r':
		case 'R':
			#pragma omp parallel for
			for (int p=0; p<num_ind; p++) {
				for (int q=0; q<num_ind; q++) {
					int i = inds[q]/n;
					int j = inds[p]/n;
					int k = inds[q]%n;
					int l = inds[p]%n;

					MULTCONJ(alpha, A[i+m*j], B[l+n*k], C[num_ind*p + q]);
				}
			}
			break;
		case 'l':
		case 'L':
			#pragma omp parallel for
			for (int p=0; p<num_ind; p++) {
				for (int q=0; q<num_ind; q++) {
					int i = inds[q]/n;
					int j = inds[p]/n;
					int k = inds[q]%n;
					int l = inds[p]%n;

					MULTCONJ(alpha, B[k+n*l], A[j+m*i], C[num_ind*p + q]);
				}
			}
			break;
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
	C: num_ind*num_ind matrix
*/
void kronI_r(const char side, const int m, const int n, const MAT_TYPE *restrict A, 
	         MAT_TYPE *restrict C, const int num_ind, const int *restrict inds) {

	__assume_aligned(A, MEM_DATA_ALIGN);
	__assume_aligned(C, MEM_DATA_ALIGN);

	switch (side) {

		default:
		case 'r':
		case 'R':
			#pragma omp parallel for
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
			#pragma omp parallel for
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
	const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
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
		const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
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

// If PRIMME lib is available define wrapper for finding eigenvalues
#if USE_PRIMME
void primme_matvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err) {

	int N = primme->n;
	MAT_TYPE *xvec;     // pointer to i-th input vector x
	MAT_TYPE *yvec;     // pointer to i-th output vector y
	MAT_TYPE *A = (MAT_TYPE *) primme->matrix;

	for (int i=0; i<*blockSize; i++) {
		xvec = (MAT_TYPE *)x + *ldx*i;
		yvec = (MAT_TYPE *)y + *ldy*i;

		#if COMPLEX
		const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
		cblas_zhemv(CblasColMajor, CblasUpper, N, &one, A, N, xvec, 1, &zero, yvec, 1);
		#else
		cblas_dsymv(CblasColMajor, CblasUpper, N, 1.0, A, N, xvec, 1, 0.0, yvec, 1);
		#endif
	}
	*err = 0;
}

/*  Finds eigenvalues and eigenvectors of A.

	A        : N*N Matrix
	N        : Size of matrix A
	evals    : pointer that will contain eigenvalues. Size should be numEvals
	evecs    : pointer that will contain eigenvectors. Size should be numEvals*N
	numEvals : number of desired eigenvectors
	initSize : number of guesses for desired eigenvectors
*/
void primmeWrapper(MAT_TYPE *A, const int N, double *evals, MAT_TYPE *evecs, const int numEvals, const int initSize) {

	primme_params primme;

	int ret;

	/* Set default values in PRIMME configuration struct */
	primme_initialize(&primme);

	/* Set problem matrix */
	primme.matrixMatvec = primme_matvec;
	primme.matrix = A;

	primme.n = N;
	primme.numEvals = numEvals;     /* Number of wanted eigenpairs */
	// primme.eps = 1e-10;             /* ||r|| <= eps * ||matrix|| */
	primme.target = primme_smallest;
	primme.initSize = initSize;

	primme_set_method(PRIMME_DYNAMIC, &primme);

	double *rnorms = mkl_malloc(primme.numEvals*sizeof(double), MEM_DATA_ALIGN);

	#if COMPLEX
	ret = zprimme(evals, (complex double *) evecs, rnorms, &primme);
	#else
	ret = dprimme(evals, evecs, rnorms, &primme);
	#endif

	if (ret != 0) {
		fprintf(primme.outputFile, 
			"Error: primme returned with nonzero exit status: %d \n", ret);
		exit(1);
	}

	mkl_free(rnorms);
	primme_free(&primme);
}

void block_matvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *err) {

	int N = primme->n;
	MAT_TYPE *xvec = x;
	MAT_TYPE *yvec = y;

	Hamil_mats *hamil_mats = primme->matrix;
	int dimSys = hamil_mats->dimSys;
	int dimEnv = hamil_mats->dimEnv;

	MAT_TYPE *temp = mkl_malloc(dimSys*dimEnv * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	#if COMPLEX
	const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	// cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &one, trans, N, op, N, &zero, temp, N);
	// cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, &one, temp, N, trans, N, &zero, newOp, N);
	#else

	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, dimSys, dimEnv, dimSys, 1.0, hamil_mats->Hsys, dimSys, xvec, dimSys, 0.0, yvec, dimSys);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimSys, dimEnv, dimEnv, 1.0, xvec, dimSys, hamil_mats->Henv, dimEnv, 1.0, yvec, dimSys);

	for (int i=0; i<hamil_mats->num_int_terms; i++) {
		cblas_dgemm(CblasColMajor, hamil_mats->trans[2*i], CblasNoTrans  , dimSys, dimEnv, dimSys, 
					1.0, hamil_mats->Hsys_ints[i], dimSys, xvec, dimSys, 0.0, temp, dimSys);
		cblas_dgemm(CblasColMajor, CblasNoTrans, hamil_mats->trans[2*i+1], dimSys, dimEnv, dimEnv, 
					hamil_mats->int_alphas[i], temp, dimSys, hamil_mats->Henv_ints[i], dimEnv, 1.0, yvec, dimSys);
	}
	
	#endif
	mkl_free(temp);

	*err = 0;
}

/*  Finds eigenvalues and eigenvectors of .

	hamil_mats : Struct containing Hamiltonian matrices

	Matvec is computed using
	H|Psi> = Hsys*Psi + Psi*Henv + Sum_i[ int_alphas[i] * Hsys_ints*Psi*Henv_ints ]
	
	N        : Size of matrix A
	evals    : pointer that will contain eigenvalues. Size should be numEvals
	evecs    : pointer that will contain eigenvectors. Size should be numEvals*N
	numEvals : number of desired eigenvectors
	initSize : number of guesses for desired eigenvectors
*/
void primmeBlockWrapper(Hamil_mats *hamil_mats, int N, double *evals, MAT_TYPE *evecs, const int numEvals, const int initSize) {

	primme_params primme;

	int ret;

	/* Set default values in PRIMME configuration struct */
	primme_initialize(&primme);

	/* Set problem matrix */
	primme.matrixMatvec = block_matvec;
	primme.matrix = hamil_mats;

	primme.n = N;
	primme.numEvals = numEvals;     /* Number of wanted eigenpairs */
	// primme.eps = 1e-10;             /* ||r|| <= eps * ||matrix|| */
	primme.target = primme_smallest;
	primme.initSize = initSize;
	primme.maxBlockSize = 1;

	primme_set_method(PRIMME_DYNAMIC, &primme);

	double *rnorms = mkl_malloc(primme.numEvals*sizeof(double), MEM_DATA_ALIGN);

	#if COMPLEX
	ret = zprimme(evals, (complex double *) evecs, rnorms, &primme);
	#else
	ret = dprimme(evals, evecs, rnorms, &primme);
	#endif

	if (ret != 0) {
		fprintf(primme.outputFile, 
			"Error: primme returned with nonzero exit status: %d \n", ret);
		exit(1);
	}

	mkl_free(rnorms);
	primme_free(&primme);
}

/* Reorder vector v so that basis can be used to guess next ground state.
*/
MAT_TYPE *reorderKron(MAT_TYPE *v, const int dimSys, const int dimEnv, const int dimSite) {

	MAT_TYPE *new_v = mkl_malloc(dimSys*dimEnv*dimSite * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	for (int i=0; i<dimSys; i++) {
		for (int j=0; j<dimEnv; j++) {
			for (int k=0; k<dimSite; k++) {
				new_v[(j*dimSys + i)*dimSite + k] = v[(j*dimSite + k)*dimSys + i];
			}
		}
	}

	return new_v;
}
#endif


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
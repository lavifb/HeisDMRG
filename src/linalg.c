#include "linalg.h"
#include <mkl.h>

/*
    \brief Compute Kronecker product of two square matrices. Sets C = alpha * kron(A,B) + C
 
    \param m 
    \param n 
    \param A m*m matrix
    \param B n*n matrix
    \param C mn*mn matrix
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

/* Print matrix from Intel MKL examples
*/
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
    int i, j;
    printf( "\n %s\n", desc );
    for(i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ ) printf( " %6.6f", a[i+j*lda] );
        printf( "\n" );
    }
}
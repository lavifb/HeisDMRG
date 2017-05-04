#include "linalg.h"
// #include "dupio.h"
#include <mkl.h>
#include <assert.h>


//________________________________________________________________________________________________________________________
///
/// \brief Compute Kronecker product of two square matrices. Sets C = alpha * kron(A,B) + C
///
/// \param m 
/// \param n 
/// \param A m*m matrix
/// \param B n*n matrix
/// \param C nm*nm matrix
///
void kron(const double alpha, const int m, const int n, const double *restrict A, const double *restrict B, double *restrict C) {
    int ldac = m*n;
    int i, j, k, l;
    for(i=0; i<m; i++) {
        for(j=0; j<m; j++) {
            for(k=0; k<n; k++) {
                for(l=0; l<n; l++) {
                    C[(n*i + k) + ldac*(n*j + l)] += alpha * A[i+m*j]*B[k+n*l];
                }
            }
        }
    }
}
/* Creates identity matrix of size N*N
*/
double *identity(const int N) {
    double *Id = mkl_calloc(N*N, sizeof(double), MEM_DATA_ALIGN);
    for(int i = 0; i < N; i++) Id[i*N+i] = 1.0;

    return Id;
}
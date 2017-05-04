#include "linalg.h"
// #include "dupio.h"
#include <mkl.h>
#include <assert.h>


//________________________________________________________________________________________________________________________
///
/// \brief Compute Kronecker product of two square matrices
///
/// \param m 
/// \param n 
/// \param A m*m matrix
/// \param B n*n matrix
///
int kron(const int m, const int n, const double *restrict A, const double *restrict B, double *restrict ret) {
	int ldac = m*n;
	int i, j, k, l;
	for(i=0; i<m; i++) {
		for(j=0; j<m; j++) {
			for(k=0; k<n; k++) {
				for(l=0; l<n; l++) {
					ret[(n*i + k) + ldac*(n*j + l)] = A[i+m*j]*B[k+n*l];
				}
			}
		}
	}
}
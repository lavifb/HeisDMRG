#include "model.h"
#include "linalg.h"
#include <mkl.h>

/*  Interaction part of Heisenberg Hamiltonian
    H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
double *HeisenH_int(const double J, const double Jz, const int dim1, const int dim2, 
                    const double *restrict Sz1, const double *restrict Sp1, 
                    const double *restrict Sz2, const double *restrict Sp2) {
    int N = dim1*dim2; // size of new basis

    double *H_int = (double *)mkl_calloc(N*N,        sizeof(double), MEM_DATA_ALIGN);
    double *Sm1   = (double *)mkl_malloc(dim1*dim1 * sizeof(double), MEM_DATA_ALIGN);
    double *Sm2   = (double *)mkl_malloc(dim2*dim2 * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(H_int,   MEM_DATA_ALIGN);
    __assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
    __assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

    // print_matrix("Sz1", dim1, dim1, Sz1, dim1);
    kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
    // print_matrix("H_int", N, N, H_int, N);

    mkl_domatcopy('C', 'c', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
    // print_matrix("Sm1", dim1, dim1, Sm1, dim1);
    mkl_domatcopy('C', 'c', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2
    // print_matrix("Sm2", dim2, dim2, Sm2, dim2);

    kron(J/2, dim1, dim2, Sp1, Sm2, H_int); // H_int += J * kron(Sp1, Sm2)
    kron(J/2, dim1, dim2, Sm1, Sp2, H_int); // H_int += J * kron(Sm1, Sp2)


    mkl_free(Sm1);
    mkl_free(Sm2);

    return H_int;
}
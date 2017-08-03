#include "hamil.h"
#include "block.h"
#include "linalg.h"
#include <mkl.h>

/*  Interaction part of Heisenberg Hamiltonian
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
double *HeisenH_int(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	double *Sz1 = block1->ops[1];
	double *Sz2 = block2->ops[1];
	double *Sp1 = block1->ops[2];
	double *Sp2 = block2->ops[2];

	int N = dim1*dim2; // size of new basis

	double *H_int = (double *)mkl_calloc(N*N,        sizeof(double), MEM_DATA_ALIGN);
	double *Sm1   = (double *)mkl_malloc(dim1*dim1 * sizeof(double), MEM_DATA_ALIGN);
	double *Sm2   = (double *)mkl_malloc(dim2*dim2 * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(H_int,   MEM_DATA_ALIGN);
	__assume_aligned(Sp1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sp2  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)

	mkl_domatcopy('C', 'C', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_domatcopy('C', 'C', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2

	// TODO: kron transpose in 1 step

	kron(J2, dim1, dim2, Sp1, Sm2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
	kron(J2, dim1, dim2, Sm1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)


	mkl_free(Sm1);
	mkl_free(Sm2);

	return H_int;
}

/*  Interaction part of Heisenberg Hamiltonian with basis restriction
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
double *HeisenH_int_r(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2,
					const int num_ind, const int *restrict inds) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	double *Sz1 = block1->ops[1];
	double *Sz2 = block2->ops[1];
	double *Sp1 = block1->ops[2];
	double *Sp2 = block2->ops[2];

	double *H_int = (double *)mkl_calloc(num_ind*num_ind, sizeof(double), MEM_DATA_ALIGN);
	double *Sm1   = (double *)mkl_malloc(dim1*dim1 * sizeof(double), MEM_DATA_ALIGN);
	double *Sm2   = (double *)mkl_malloc(dim2*dim2 * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(H_int,   MEM_DATA_ALIGN);
	__assume_aligned(Sp1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sp2  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron_r(Jz, dim1, dim2, Sz1, Sz2, H_int, num_ind, inds); // H_int += Jz * kron(Sz1, Sz2)

	mkl_domatcopy('C', 'C', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_domatcopy('C', 'C', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2

	// TODO: kron transpose in 1 step

	kron_r(J2, dim1, dim2, Sp1, Sm2, H_int, num_ind, inds); // H_int += J/2 * kron(Sp1, Sm2)
	kron_r(J2, dim1, dim2, Sm1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sm1, Sp2)


	mkl_free(Sm1);
	mkl_free(Sm2);

	return H_int;
}
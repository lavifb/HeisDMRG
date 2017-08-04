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

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)

	kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
	kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)

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

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron_r(Jz, dim1, dim2, Sz1, Sz2, H_int, num_ind, inds); // H_int += Jz * kron(Sz1, Sz2)

	kronT_r('r', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sp1, Sm2)
	kronT_r('l', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sm1, Sp2)

	return H_int;
}
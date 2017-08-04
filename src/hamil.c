#include "hamil.h"
#include "block.h"
#include "linalg.h"
#include <mkl.h>

/*  Interaction part of Heisenberg Hamiltonian
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
MAT_TYPE *HeisenH_int(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	MAT_TYPE *Sz1 = block1->ops[1];
	MAT_TYPE *Sz2 = block2->ops[1];
	MAT_TYPE *Sp1 = block1->ops[2];
	MAT_TYPE *Sp2 = block2->ops[2];

	int N = dim1*dim2; // size of new basis

	MAT_TYPE *H_int = (MAT_TYPE *)mkl_calloc(N*N,        sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *Sm1   = (MAT_TYPE *)mkl_malloc(dim1*dim1 * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *Sm2   = (MAT_TYPE *)mkl_malloc(dim2*dim2 * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	__assume_aligned(H_int,   MEM_DATA_ALIGN);
	__assume_aligned(Sp1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sp2  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)

	#if COMPLEX
	MKL_Complex16 one = {.real=1.0, .imag=0.0};
	mkl_zomatcopy('C', 'C', dim1, dim1, one, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_zomatcopy('C', 'C', dim2, dim2, one, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2
	#else
	mkl_domatcopy('C', 'C', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_domatcopy('C', 'C', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2
	#endif
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
MAT_TYPE *HeisenH_int_r(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2,
					const int num_ind, const int *restrict inds) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	MAT_TYPE *Sz1 = block1->ops[1];
	MAT_TYPE *Sz2 = block2->ops[1];
	MAT_TYPE *Sp1 = block1->ops[2];
	MAT_TYPE *Sp2 = block2->ops[2];

	MAT_TYPE *H_int = (MAT_TYPE *)mkl_calloc(num_ind*num_ind, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *Sm1   = (MAT_TYPE *)mkl_malloc(dim1*dim1 * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *Sm2   = (MAT_TYPE *)mkl_malloc(dim2*dim2 * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	__assume_aligned(H_int,   MEM_DATA_ALIGN);
	__assume_aligned(Sp1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sp2  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron_r(Jz, dim1, dim2, Sz1, Sz2, H_int, num_ind, inds); // H_int += Jz * kron(Sz1, Sz2)

	#if COMPLEX
	MKL_Complex16 one = {.real=1.0, .imag=0.0};
	mkl_zomatcopy('C', 'C', dim1, dim1, one, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_zomatcopy('C', 'C', dim2, dim2, one, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2
	#else
	mkl_domatcopy('C', 'C', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_domatcopy('C', 'C', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2
	#endif
	// TODO: kron transpose in 1 step

	kron_r(J2, dim1, dim2, Sp1, Sm2, H_int, num_ind, inds); // H_int += J/2 * kron(Sp1, Sm2)
	kron_r(J2, dim1, dim2, Sm1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sm1, Sp2)


	mkl_free(Sm1);
	mkl_free(Sm2);

	return H_int;
}
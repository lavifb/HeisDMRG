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

	MAT_TYPE *H_int = (MAT_TYPE *)mkl_calloc(N*N, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

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
MAT_TYPE *HeisenH_int_r(const double* H_params, const DMRGBlock *block1, const DMRGBlock *block2,
					const int num_ind, const int *restrict inds) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	MAT_TYPE *Sz1 = block1->ops[1];
	MAT_TYPE *Sz2 = block2->ops[1];
	MAT_TYPE *Sp1 = block1->ops[2];
	MAT_TYPE *Sp2 = block2->ops[2];

	MAT_TYPE *H_int = (MAT_TYPE *)mkl_calloc(num_ind*num_ind, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron_r(Jz, dim1, dim2, Sz1, Sz2, H_int, num_ind, inds); // H_int += Jz * kron(Sz1, Sz2)
	kronT_r('r', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sp1, Sm2)
	kronT_r('l', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sm1, Sp2)

	return H_int;
}

Hamil_mats *HeisenH_int_mats(double *H_params, const DMRGBlock *block1, const DMRGBlock *block2) {

	Hamil_mats *hamil_mats = mkl_malloc(sizeof(Hamil_mats), MEM_DATA_ALIGN);

	hamil_mats->dimSys = block1->d_block;
	hamil_mats->dimEnv = block2->d_block;
	hamil_mats->Hsys = block1->ops[0];
	hamil_mats->Henv = block2->ops[0];
	hamil_mats->num_int_terms = 3;
	hamil_mats->int_alphas = H_params;
	
	// Set the right trans array
	hamil_mats->trans = mkl_malloc(6 * sizeof(CBLAS_TRANSPOSE), MEM_DATA_ALIGN);
	hamil_mats->trans[0] = CblasNoTrans;
	hamil_mats->trans[1] = CblasTrans;
	hamil_mats->trans[2] = CblasTrans;
	hamil_mats->trans[3] = CblasNoTrans;
	hamil_mats->trans[4] = CblasNoTrans;
	hamil_mats->trans[5] = CblasNoTrans;

	// Sys interaction mats
	hamil_mats->Hsys_ints = mkl_malloc(3 * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	hamil_mats->Hsys_ints[0] = block1->ops[2];
	hamil_mats->Hsys_ints[1] = block1->ops[2];
	hamil_mats->Hsys_ints[2] = block1->ops[1];

	// Env interaction mats
	hamil_mats->Henv_ints = mkl_malloc(3 * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	hamil_mats->Henv_ints[0] = block1->ops[2];
	hamil_mats->Henv_ints[1] = block1->ops[2];
	hamil_mats->Henv_ints[2] = block1->ops[1];

	return hamil_mats;
}

void freeHamil_mats(Hamil_mats *hamil_mats) {

	if (hamil_mats->trans) { mkl_free(hamil_mats->trans); }
	if (hamil_mats->Hsys_ints) { mkl_free(hamil_mats->Hsys_ints); }
	if (hamil_mats->Henv_ints) { mkl_free(hamil_mats->Henv_ints); }
	mkl_free(hamil_mats);
}
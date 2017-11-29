#include "hamil.h"
#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>
#include <string.h>

/*  Wrapper function to find lowest energy states. It uses the faster PRIMME library if available

	sys_enl : enlarged sys block
	env_enl : enlarged env block
	model   : model containing sim parameters
	num_restr_ind    : number of restricted basis inds for restricting basis. Set to -1 to not restrict basis
	restr_basis_inds : restricted basis inds for restricting basis
	num_states  : number of states being searched for
	psi0_guessp : pointer to potential ground state guesses
	energies : pointer to output which gives energies

	returns  : ground state
*/
MAT_TYPE *getLowestEStates(const DMRGBlock *sys_enl, const DMRGBlock *env_enl, const model_t* model, int num_restr_ind,
	const int *restr_basis_inds, int num_states, MAT_TYPE **psi0_guessp, double *energies) {

	// Use the faster PRIMME library if available. Otherwise, default to LAPACK.
	#if USE_PRIMME
		int dimSup = sys_enl->d_block * env_enl->d_block;

		// Setup ground state guess
		MAT_TYPE *psi0 = mkl_malloc(dimSup * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		int numGuesses = 0;
		if (psi0_guessp != NULL && *psi0_guessp != NULL) {
			memcpy(psi0, *psi0_guessp, dimSup * sizeof(MAT_TYPE));
			numGuesses = 1;
		}

		hamil_mats_t *hamils_mats = model->H_int_mats(model, sys_enl, env_enl);
		primmeBlockWrapper(hamils_mats, dimSup, energies, psi0, num_states, numGuesses);
		freehamil_mats_t(hamils_mats);

		MAT_TYPE *psi0_r = psi0;
		if (num_restr_ind >= 0) {
			psi0_r = restrictVec(psi0, num_restr_ind, restr_basis_inds);
			mkl_free(psi0);
		}
	#else
		if (num_restr_ind < 0) {
			num_restr_ind = sys_enl->d_block * env_enl->d_block;
			restr_basis_inds = mkl_malloc(num_restr_ind * sizeof(int), MEM_DATA_ALIGN);
			for (int i=0; i<num_restr_ind; i++) { restr_basis_inds[i] = i; }
		}

		// Restricted Superblock Hamiltonian
		MAT_TYPE *Hs_r = model->H_int_r(model, sys_enl, env_enl, num_restr_ind, restr_basis_inds);
		kronI_r('R', dimSys, dimEnv, sys_enl->ops[0], Hs_r, num_restr_ind, restr_basis_inds);
		kronI_r('L', dimSys, dimEnv, env_enl->ops[0], Hs_r, num_restr_ind, restr_basis_inds);

		// Setup ground state guess
		MAT_TYPE *psi0_r;
		int numGuesses = 0;
		if (psi0_guessp != NULL && *psi0_guessp != NULL) {
			psi0_r  = restrictVec(*psi0_guessp, num_restr_ind, restr_basis_inds);
			numGuesses = 1;
		} else {
			psi0_r = mkl_malloc(num_restr_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		}

		int info = 0;
		int num_es_found;
		int *isuppz = mkl_malloc(2 * sizeof(int), MEM_DATA_ALIGN);

		#if COMPLEX
		info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, num_states, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#else
		info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, num_states, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#endif

		if (info > 0) {
			printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
			exit(1);
		}
		mkl_free(isuppz);
		mkl_free(Hs_r);
		if (num_restr_ind < 0) { mkl_free(restr_basis_inds); }
	#endif

	return psi0_r;
}


/*  Interaction part of Heisenberg Hamiltonian
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
MAT_TYPE *HeisenH_int(const model_t* model, const DMRGBlock *block1, const DMRGBlock *block2) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	MAT_TYPE *Sz1 = block1->ops[1];
	MAT_TYPE *Sz2 = block2->ops[1];
	MAT_TYPE *Sp1 = block1->ops[2];
	MAT_TYPE *Sp2 = block2->ops[2];

	int N = dim1*dim2; // size of new basis

	MAT_TYPE *H_int = mkl_calloc(N*N, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	double *H_params = model->H_params;
	double J2 = H_params[0]/2; // J/2
	double Jz = H_params[1];

	kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
	kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
	kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)

	return H_int;
}

/*  Interaction part of Heisenberg Hamiltonian with basis restriction
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
MAT_TYPE *HeisenH_int_r(const model_t* model, const DMRGBlock *block1, const DMRGBlock *block2, 
	const int num_ind, const int *restrict inds) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	MAT_TYPE *Sz1 = block1->ops[1];
	MAT_TYPE *Sz2 = block2->ops[1];
	MAT_TYPE *Sp1 = block1->ops[2];
	MAT_TYPE *Sp2 = block2->ops[2];

	MAT_TYPE *H_int = mkl_calloc(num_ind*num_ind, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	double *H_params = model->H_params;
	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron_r(Jz, dim1, dim2, Sz1, Sz2, H_int, num_ind, inds); // H_int += Jz * kron(Sz1, Sz2)
	kronT_r('r', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sp1, Sm2)
	kronT_r('l', J2, dim1, dim2, Sp1, Sp2, H_int, num_ind, inds); // H_int += J/2 * kron(Sm1, Sp2)

	return H_int;
}

hamil_mats_t *HeisenH_int_mats(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2) {

	hamil_mats_t *hamil_mats = mkl_malloc(sizeof(hamil_mats_t), MEM_DATA_ALIGN);

	hamil_mats->dimSys = block1->d_block;
	hamil_mats->dimEnv = block2->d_block;
	hamil_mats->Hsys = block1->ops[0];
	hamil_mats->Henv = block2->ops[0];
	hamil_mats->num_int_terms = 3;

	// Coefs from H_params
	double *H_params = model->H_params;
	hamil_mats->int_alphas = mkl_malloc(3 * sizeof(int), MEM_DATA_ALIGN);
	hamil_mats->int_alphas[0] = H_params[0]/2;
	hamil_mats->int_alphas[1] = H_params[0]/2;
	hamil_mats->int_alphas[2] = H_params[1];
	
	// Set the right trans array
	// Note: For the right matvec calculation you need an extra transpose on one side of the calculation.
	//       Here we chose the left side (even indexes) to have an extra transpose.
	hamil_mats->trans = mkl_malloc(6 * sizeof(CBLAS_TRANSPOSE), MEM_DATA_ALIGN);
	hamil_mats->trans[0] = CblasTrans;
	hamil_mats->trans[1] = CblasTrans;
	hamil_mats->trans[2] = CblasNoTrans;
	hamil_mats->trans[3] = CblasNoTrans;
	hamil_mats->trans[4] = CblasTrans;
	hamil_mats->trans[5] = CblasNoTrans;

	// Sys interaction mats
	hamil_mats->Hsys_ints = mkl_malloc(3 * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	hamil_mats->Hsys_ints[0] = block1->ops[2];
	hamil_mats->Hsys_ints[1] = block1->ops[2];
	hamil_mats->Hsys_ints[2] = block1->ops[1];

	// Env interaction mats
	hamil_mats->Henv_ints = mkl_malloc(3 * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	hamil_mats->Henv_ints[0] = block2->ops[2];
	hamil_mats->Henv_ints[1] = block2->ops[2];
	hamil_mats->Henv_ints[2] = block2->ops[1];

	return hamil_mats;
}

/*  Interaction part of Ladder Heisenberg Hamiltonian
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2) for each connection
*/
MAT_TYPE *LadderH_int(const model_t* model, const DMRGBlock *block1, const DMRGBlock *block2) {

	int dim1 = block1->d_block;
	int dim2 = block2->d_block;

	int N = dim1*dim2; // size of new basis

	MAT_TYPE *H_int = mkl_calloc(N*N, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	double *H_params = model->H_params;
	double J2 = H_params[0]/2; // J/2
	double Jz = H_params[1];

	int lw = model->ladder_width;

	// Connections up the ladder
	for (int i=0; i<lw; i++) {
		MAT_TYPE *Sz1 = block1->ops[2*i+1];
		MAT_TYPE *Sz2 = block2->ops[2*i+1];
		MAT_TYPE *Sp1 = block1->ops[2*i+2];
		MAT_TYPE *Sp2 = block2->ops[2*i+2];

		kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
		kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
		kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)
	}

	// Check for full width ladder
	if (block1->length%lw != 0) {

		int conn_i = (block1->length-1)%lw;
		// check if snaking is going up or down
		if ((block1->length-1)/lw % 2 == 1) {
			conn_i = lw - 1 - conn_i;

			// Periodic boundary if snaking down
			MAT_TYPE *Sz1 = block1->ops[2*(lw-1)+1];
			MAT_TYPE *Sz2 = block2->ops[1];
			MAT_TYPE *Sp1 = block1->ops[2*(lw-1)+2];
			MAT_TYPE *Sp2 = block2->ops[2];

			kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
			kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
			kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)

			// Connecting piece if snaking down
			MAT_TYPE *Sz1 = block1->ops[2*conn_i+1];
			MAT_TYPE *Sz2 = block2->ops[2*(conn_i-1)+1];
			MAT_TYPE *Sp1 = block1->ops[2*conn_i+2];
			MAT_TYPE *Sp2 = block2->ops[2*(conn_i-1)+2];
	
			kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
			kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
			kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)

		} else {
			// Periodic boundary if snaking up
			MAT_TYPE *Sz1 = block1->ops[1];
			MAT_TYPE *Sz2 = block2->ops[2*(lw-1)+1];
			MAT_TYPE *Sp1 = block1->ops[2];
			MAT_TYPE *Sp2 = block2->ops[2*(lw-1)+2];

			kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
			kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
			kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)

			// Connecting piece if snaking up
			MAT_TYPE *Sz1 = block1->ops[2*conn_i+1];
			MAT_TYPE *Sz2 = block2->ops[2*(conn_i+1)+1];
			MAT_TYPE *Sp1 = block1->ops[2*conn_i+2];
			MAT_TYPE *Sp2 = block2->ops[2*(conn_i+1)+2];
	
			kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)
			kronT('r', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
			kronT('l', J2, dim1, dim2, Sp1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)
		}
	}

	return H_int;
}

/*  Heisenberg interaction in ladder system.
*/
hamil_mats_t *LadderH_int_mats(const model_t *model, const DMRGBlock *block1, const DMRGBlock *block2) {

	hamil_mats_t *hamil_mats = mkl_malloc(sizeof(hamil_mats_t), MEM_DATA_ALIGN);

	hamil_mats->dimSys = block1->d_block;
	hamil_mats->dimEnv = block2->d_block;
	hamil_mats->Hsys = block1->ops[0];
	hamil_mats->Henv = block2->ops[0];

	int lw = model->ladder_width;
	int num_int_terms = 3*lw;
	// Check for full width ladder
	if (block1->length%lw != 0) {
		num_int_terms += 3*2;
	}
	hamil_mats->num_int_terms = num_int_terms;

	// Coefs from H_params
	double *H_params = model->H_params;
	hamil_mats->int_alphas = mkl_malloc(num_int_terms * sizeof(int), MEM_DATA_ALIGN);
	for (int i=0; i<num_int_terms/3; i++) {
		hamil_mats->int_alphas[i*3 + 0] = H_params[0]/2;
		hamil_mats->int_alphas[i*3 + 1] = H_params[0]/2;
		hamil_mats->int_alphas[i*3 + 2] = H_params[1];
	}
	
	// Set the right trans array
	// Note: For the right matvec calculation you need an extra transpose on one side of the calculation.
	//       Here we chose the left side (even indexes) to have an extra transpose.
	hamil_mats->trans = mkl_malloc(2*num_int_terms * sizeof(CBLAS_TRANSPOSE), MEM_DATA_ALIGN);
	for (int i=0; i<num_int_terms/6; i++) {
		hamil_mats->trans[i*3 + 0] = CblasTrans;
		hamil_mats->trans[i*3 + 1] = CblasTrans;
		hamil_mats->trans[i*3 + 2] = CblasNoTrans;
		hamil_mats->trans[i*3 + 3] = CblasNoTrans;
		hamil_mats->trans[i*3 + 4] = CblasTrans;
		hamil_mats->trans[i*3 + 5] = CblasNoTrans;
	}

	// Sys interaction mats for across connections
	hamil_mats->Hsys_ints = mkl_malloc(num_int_terms * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i=0; i<lw; i++) {
		hamil_mats->Hsys_ints[3*i]     = block1->ops[2*i + 2];
		hamil_mats->Hsys_ints[3*i + 1] = block1->ops[2*i + 2];
		hamil_mats->Hsys_ints[3*i + 2] = block1->ops[2*i + 1];
	}

	// Env interaction mats for across connections
	hamil_mats->Henv_ints = mkl_malloc(num_int_terms * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	for (int i=0; i<lw; i++) {
		hamil_mats->Henv_ints[3*i]     = block2->ops[2*i + 2];
		hamil_mats->Henv_ints[3*i + 1] = block2->ops[2*i + 2];
		hamil_mats->Henv_ints[3*i + 2] = block2->ops[2*i + 1];
	}

	if (block1->length%lw != 0) {

		int conn_i = (block1->length-1)%lw;
		// check if snaking is going up or down
		if ((block1->length-1)/lw % 2 == 1) {
			conn_i = lw - 1 - conn_i;

			// Periodic boundary if snaking down
			hamil_mats->Hsys_ints[3*lw]     = block1->ops[2*(lw-1)+2];
			hamil_mats->Hsys_ints[3*lw + 1] = block1->ops[2*(lw-1)+2];
			hamil_mats->Hsys_ints[3*lw + 2] = block1->ops[2*(lw-1)+1];
			hamil_mats->Henv_ints[3*lw]     = block2->ops[2];
			hamil_mats->Henv_ints[3*lw + 1] = block2->ops[2];
			hamil_mats->Henv_ints[3*lw + 2] = block2->ops[1];

			// Connecting piece if snaking down
			hamil_mats->Hsys_ints[3*(lw+1)]     = block1->ops[2*conn_i+2];
			hamil_mats->Hsys_ints[3*(lw+1) + 1] = block1->ops[2*conn_i+2];
			hamil_mats->Hsys_ints[3*(lw+1) + 2] = block1->ops[2*conn_i+1];
			hamil_mats->Henv_ints[3*(lw+1)]     = block2->ops[2*(conn_i-1)+2];
			hamil_mats->Henv_ints[3*(lw+1) + 1] = block2->ops[2*(conn_i-1)+2];
			hamil_mats->Henv_ints[3*(lw+1) + 2] = block2->ops[2*(conn_i-1)+1];

		} else {
			// Periodic boundary if snaking up
			hamil_mats->Hsys_ints[3*lw]     = block1->ops[2];
			hamil_mats->Hsys_ints[3*lw + 1] = block1->ops[2];
			hamil_mats->Hsys_ints[3*lw + 2] = block1->ops[1];
			hamil_mats->Henv_ints[3*lw]     = block2->ops[2*(lw-1)+2];
			hamil_mats->Henv_ints[3*lw + 1] = block2->ops[2*(lw-1)+2];
			hamil_mats->Henv_ints[3*lw + 2] = block2->ops[2*(lw-1)+1];

			// Connecting piece if snaking up
			hamil_mats->Hsys_ints[3*(lw+1)]     = block1->ops[2*conn_i+2];
			hamil_mats->Hsys_ints[3*(lw+1) + 1] = block1->ops[2*conn_i+2];
			hamil_mats->Hsys_ints[3*(lw+1) + 2] = block1->ops[2*conn_i+1];
			hamil_mats->Henv_ints[3*(lw+1)]     = block2->ops[2*(conn_i+1)+2];
			hamil_mats->Henv_ints[3*(lw+1) + 1] = block2->ops[2*(conn_i+1)+2];
			hamil_mats->Henv_ints[3*(lw+1) + 2] = block2->ops[2*(conn_i+1)+1];
		}

	}

	return hamil_mats;
}

void freehamil_mats_t(hamil_mats_t *hamil_mats) {

	if (hamil_mats->int_alphas) { mkl_free(hamil_mats->int_alphas); }
	if (hamil_mats->trans) { mkl_free(hamil_mats->trans); }
	if (hamil_mats->Hsys_ints) { mkl_free(hamil_mats->Hsys_ints); }
	if (hamil_mats->Henv_ints) { mkl_free(hamil_mats->Henv_ints); }
	mkl_free(hamil_mats);
}
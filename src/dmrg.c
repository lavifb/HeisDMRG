#include "dmrg.h"
#include "block.h"
#include "model.h"
#include "hamil.h"
#include "sector.h"
#include "meas.h"
#include "linalg.h"
#include "logio.h"
#include "matio.h"
#include "uthash.h"
#include <mkl.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

/* Single DMRG step
   
   m: truncation dimension size

   psi0_guessp: pointer to guess for psi0. Calculated psi0 is returned in theis pointer.
                Set  psi0_guessp = NULL to not use eigenstate guessing and not return eigenstate.
                Set *psi0_guessp = NULL to not use eigenstate guessing but return eigenstate for future guessing.

   returns enlarged system block
*/
DMRGBlock *single_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz, MAT_TYPE **const psi0_guessp) {

	DMRGBlock *sys_enl, *env_enl;
	sector_t *sys_enl_sectors, *env_enl_sectors;
	model_t *model = sys->model;

	sys_enl = enlargeBlock(sys);
	sys_enl_sectors = sectorize(sys_enl);
	if (sys == env) { // Don't recalculate
		env_enl = sys_enl;
		env_enl_sectors = sys_enl_sectors;
	}
	else {
		env_enl = enlargeBlock(env);
		env_enl_sectors = sectorize(env_enl);
	}

	int dimSys = sys_enl->d_block;
	int dimEnv = env_enl->d_block;
	int dimSup = dimSys * dimEnv;

	// indexes used for restricting Hs
	int *restr_basis_inds = mkl_malloc(dimSup * sizeof(int), MEM_DATA_ALIGN);
	int num_restr_ind;

	// Get restricted basis
	// sup_sectors stores sectors for superblock
	sector_t *sup_sectors = getRestrictedBasis(sys_enl_sectors, env_enl_sectors, target_mz, dimEnv, &num_restr_ind, restr_basis_inds);


	// Restricted Superblock Hamiltonian
	MAT_TYPE *Hs_r = model->H_int_r(model->H_params, sys_enl, env_enl, num_restr_ind, restr_basis_inds);
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

	// Find ground state
	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	// TODO: move this to its own function.
	// Use the faster PRIMME library if available. Otherwise, default to LAPACK.
	#if USE_PRIMME
		primmeWrapper(Hs_r, num_restr_ind, energies, psi0_r, 1, numGuesses);
	#else
		int info = 0;
		int num_es_found;
		int *isuppz = mkl_malloc(2 * sizeof(int), MEM_DATA_ALIGN);

		#if COMPLEX
		info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, 1, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#else
		info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, 1, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#endif

		if (info > 0) {
			printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
			exit(1);
		}
		mkl_free(isuppz);
	#endif

	mkl_free(Hs_r);

	sys_enl->energy = energies[0]; // record ground state energy
	mkl_free(energies);

	// Transformation Matrix
	int mm = (dimSys < m) ? dimSys : m; // use min(dimSys, m) 
	MAT_TYPE *trans_full = mkl_calloc(dimSys*dimSys, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	// Eigenvalues of density matrix
	int lamb_i = 0;
	double *lambs = mkl_malloc(dimSys * sizeof(double), MEM_DATA_ALIGN);

	// state mzs to eventually truncate and put into sys_enl->mzs
	int *sys_mzs_full = mkl_malloc(dimSys * sizeof(int), MEM_DATA_ALIGN);

	// Loop over sectors to find what basis inds to keep
	for (sector_t *sec=sup_sectors; sec != NULL; sec=sec->hh.next) {
		int mz = sec->id;
		// printf("mz = %d\n", mz);
		int env_mz = target_mz - mz;
		int n_sec = sec->num_ind;

		sector_t *sys_enl_mz, *env_enl_mz;
		HASH_FIND_INT(sys_enl_sectors, &mz    , sys_enl_mz);
		HASH_FIND_INT(env_enl_sectors, &env_mz, env_enl_mz);
		assert(sys_enl_mz != NULL);
		// SOMETHING WRONG HERE!!! (MAYBE??)
		if (env_enl_mz == NULL) {
			continue;
		}
		int dimSys_sec = sys_enl_mz->num_ind;
		int dimEnv_sec = env_enl_mz->num_ind;
		assert(dimSys_sec * dimEnv_sec == n_sec);

		// target states
		int num_targets = 1;
		MAT_TYPE **targets = mkl_malloc(num_targets * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

		// define target states
		targets[0] = restrictVec(psi0_r, n_sec, sec->inds); // ground state
		// targets[1] = restrictVec(psi_r, n_sec, sec->inds); // tracked state

		// Density matrix rho_sec
		MAT_TYPE *rho_sec = mkl_calloc(dimSys_sec*dimSys_sec, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

		// target state needs to be arranged as a dimSys * dimEnv to trace out env
		// Put sys_basis on rows and env_basis on the cols by taking transpose
		// To not take transpose twice, just take conj and take conjTrans on left side of dgemm bellow

		// set weights for target states to be equal
		const double alpha = 1.0/num_targets;
		#if COMPLEX
		const MKL_Complex16 one = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zalpha = {.real=alpha, .imag=0.0};
		#else
		#endif

		for (int i=0; i<num_targets; i++) {

			#if COMPLEX
			mkl_zimatcopy('C', 'R', dimEnv_sec, dimSys_sec, one, targets[i], dimEnv_sec, dimEnv_sec);
			// Trace out Environment to make rho (Note transpose structure as described above)
			cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimSys_sec, dimSys_sec, dimEnv_sec, 
						&zalpha, targets[i], dimEnv_sec, targets[i], dimEnv_sec, &one, rho_sec, dimSys_sec);
			#else
			// Trace out Environment to make rho (No conjugation needed here)
			cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimSys_sec, dimSys_sec, dimEnv_sec, 
						alpha, targets[i], dimEnv_sec, targets[i], dimEnv_sec, 1.0, rho_sec, dimSys_sec);
			#endif

			mkl_free(targets[i]);
		}

		mkl_free(targets);

		// diagonalize rho_sec and add to list of eigenvalues
		// LAPACK faster since we need many eigenvalues
		int mm_sec = (dimSys_sec < mm) ? dimSys_sec : mm;
		MAT_TYPE *trans_sec = (MAT_TYPE *)mkl_malloc(dimSys_sec*mm_sec * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		int *isuppz_sec = (int *)mkl_malloc(2*dimSys_sec * sizeof(int), MEM_DATA_ALIGN);
		int num_es_found;
		assert(lamb_i + mm_sec - 1 < dimSys);

		#if COMPLEX
		int info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', dimSys_sec, rho_sec, dimSys_sec, 0.0, 0.0,
				dimSys_sec-mm_sec+1, dimSys_sec, 0.0, &num_es_found, &lambs[lamb_i], trans_sec, dimSys_sec, isuppz_sec);
		#else
		int info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', dimSys_sec, rho_sec, dimSys_sec, 0.0, 0.0,
				dimSys_sec-mm_sec+1, dimSys_sec, 0.0, &num_es_found, &lambs[lamb_i], trans_sec, dimSys_sec, isuppz_sec);
		#endif

		if (info > 0) {
			printf("Failed to find eigenvalues of density matrix\n");
			exit(1);
		}
		mkl_free(rho_sec);
		mkl_free(isuppz_sec);

		// copy trans_sec into trans using the proper basis
		for (int i = 0; i < mm_sec; i++) {
			for (int j = 0; j < dimSys_sec; j++) {
				// copy value using proper index basis
				trans_full[lamb_i*dimSys + sys_enl_mz->inds[j]] = trans_sec[i*dimSys_sec + j];
			}
			// keep track of mzs for the enlarged block
			sys_mzs_full[lamb_i] = mz;
			lamb_i++;
		}
		
		mkl_free(trans_sec);
	}

	freeSectors(sup_sectors);

	// Some dimensions may already be dropped
	int newDimSys = lamb_i;
	assert(newDimSys <= dimSys);

	MAT_TYPE *trans = (MAT_TYPE *)mkl_malloc(dimSys*mm * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	assert(mm <= newDimSys);
	int *sorted_inds = dsort2(newDimSys, lambs);

	// copy to trans in right order
	for (int i = 0; i < mm; i++) {
		memcpy(&trans[i*dimSys], &trans_full[sorted_inds[i]*dimSys], dimSys * sizeof(MAT_TYPE));
		sys_enl->mzs[i] = sys_mzs_full[sorted_inds[i]];
	}

	mkl_free(sys_mzs_full);
	mkl_free(trans_full);
	
	double truncation_err = 1.0;
	for (int i = 0; i < mm; i++) {
		truncation_err -= lambs[i];
	}
	sys_enl->trunc_err = truncation_err;
	mkl_free(lambs);
	mkl_free(sorted_inds);

	// Transform operators into new basis
	transformOps(sys_enl->num_ops, dimSys, mm, trans, sys_enl->ops);
	sys_enl->d_block = mm; // set block basis size to transformed value
	sys_enl->d_trans = dimSys;
	sys_enl->trans = trans;

	// realloc and return psi0_guessp for later guess
	if (psi0_guessp != NULL) {
		// full psi0 for later eigenstate prediction
		MAT_TYPE *psi0 = unrestrictVec(dimSup, psi0_r, num_restr_ind, restr_basis_inds);

		if (*psi0_guessp == NULL) {
			*psi0_guessp = mkl_malloc(mm*dimEnv * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		} else {
			// Check overlap of guess and calculated eigenstate
			// #define PRINT_OVERLAP
			#ifdef PRINT_OVERLAP

				#if COMPLEX
				complex double zoverlap;
				cblas_zdotc_sub(dimSup, psi0, 1, *psi0_guessp, 1, &zoverlap);
				double overlap = cabs(zoverlap);
				#else
				double overlap = fabs(cblas_ddot(dimSup, psi0, 1, *psi0_guessp, 1));
				// overlap /= cblas_dnrm2(dimSup, psi0, 1) * cblas_dnrm2(dimSup, *psi0_guessp, 1);
				#endif
				printf("Overlap <psi0|psi0_guess> = %.8f\n", overlap);

			#endif

			// if (overlap < .9) {
				// printf("Guess is bad!!\n");
				// print_matrix("psi0_guess", dimEnv, dimSys, *psi0_guessp, dimEnv);
				// print_matrix("psi0"      , dimEnv, dimSys, psi0        , dimEnv);

				// printf("\ndimSys = %3d  dimEnv = %3d  dimSup = %3d\n", dimSys, dimEnv, dimSup);
				// exit(1);
			// }

			*psi0_guessp = mkl_realloc(*psi0_guessp, mm*dimEnv * sizeof(MAT_TYPE));
		}

		#if COMPLEX
		const MKL_Complex16 one = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, mm, dimEnv, dimSys, 
						&one, trans, dimSys, psi0, dimEnv, &zero, *psi0_guessp, mm);
		#else
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasTrans, mm, dimEnv, dimSys, 
						1.0, trans, dimSys, psi0, dimEnv, 0.0, *psi0_guessp, mm);
		#endif
		mkl_free(psi0);
	}

	mkl_free(restr_basis_inds);
	mkl_free(psi0_r);
	freeSectors(sys_enl_sectors);
	// Free enlarged environment block
	if (sys != env) {
		freeDMRGBlock(env_enl);
		freeSectors(env_enl_sectors);
	}

	return sys_enl;
}
/* DMRG step that records measurements.
   Use on the last half sweep to measure operators as the system builds.
   
   m: truncation dimension size

   returns enlarged system block
*/
meas_data_t *meas_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz, MAT_TYPE **const psi0_guessp) {

	DMRGBlock *sys_enl, *env_enl;
	sector_t *sys_enl_sectors, *env_enl_sectors;
	model_t *model = sys->model;

	sys_enl = enlargeBlock(sys);
	sys_enl_sectors = sectorize(sys_enl);
	if (sys == env) { // Don't recalculate
		env_enl = sys_enl;
		env_enl_sectors = sys_enl_sectors;
	}
	else {
		env_enl = enlargeBlock(env);
		env_enl_sectors = sectorize(env_enl);
	}

	int dimSys = sys_enl->d_block;
	int dimEnv = env_enl->d_block;
	int dimSup = dimSys * dimEnv;

	// indexes used for restricting Hs
	int *restr_basis_inds = mkl_malloc(dimSup * sizeof(int), MEM_DATA_ALIGN);
	int num_restr_ind;

	// Get restricted basis
	// sup_sectors stores sectors for superblock
	sector_t *sup_sectors = getRestrictedBasis(sys_enl_sectors, env_enl_sectors, target_mz, dimEnv, &num_restr_ind, restr_basis_inds);
	freeSectors(sup_sectors);
	freeSectors(sys_enl_sectors);

	// RestrictedSuperblock Hamiltonian
	MAT_TYPE *Hs_r = model->H_int_r(model->H_params, sys_enl, env_enl, num_restr_ind, restr_basis_inds);
	kronI_r('R', dimSys, dimEnv, sys_enl->ops[0], Hs_r, num_restr_ind, restr_basis_inds);
	kronI_r('L', dimSys, dimEnv, env_enl->ops[0], Hs_r, num_restr_ind, restr_basis_inds);

	// Free enlarged environment block
	if (sys != env) {
		freeDMRGBlock(env_enl);
		freeSectors(env_enl_sectors);
	}

	// Setup ground state guess
	MAT_TYPE *psi0_r;
	int numGuesses = 0;
	if (psi0_guessp != NULL && *psi0_guessp != NULL) {
		psi0_r  = restrictVec(*psi0_guessp, num_restr_ind, restr_basis_inds);
		numGuesses = 1;
	} else {
		psi0_r = (MAT_TYPE *)mkl_malloc(num_restr_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	}

	// Find ground state
	double *energies = (double *)mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	// Use the faster PRIMME library if available. Otherwise, default to LAPACK.
	#if USE_PRIMME
		primmeWrapper(Hs_r, num_restr_ind, energies, psi0_r, 1, numGuesses);
	#else
		int info = 0;
		int num_es_found;
		int *isuppz = (int *)mkl_malloc(2 * sizeof(int), MEM_DATA_ALIGN);

		#if COMPLEX
		info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, 1, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#else
		info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
				0.0, 0.0, 1, 1, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, isuppz);
		#endif

		if (info > 0) {
			printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
			exit(1);
		}
		mkl_free(isuppz);
	#endif

	mkl_free(Hs_r);

	meas_data_t *meas = createMeas(sys_enl->num_ops - model->num_ops);
	meas->energy = energies[0] / (sys_enl->length + env_enl->length);
	mkl_free(energies);

	// Make Measurements

	// <S_i> spins
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* supOp_r = (MAT_TYPE *)mkl_calloc(num_restr_ind*num_restr_ind, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		kronI_r('R', dimSys, dimEnv, sys_enl->ops[i + model->num_ops], supOp_r, num_restr_ind, restr_basis_inds);

		transformOps(1, num_restr_ind, 1, psi0_r, &supOp_r);

		#if COMPLEX
		meas->Szs[i] = (*supOp_r).real;
		#else
		meas->Szs[i] = *supOp_r;
		#endif

		mkl_free(supOp_r);
	}

	// <S_i S_j> correlations
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* SSop = (MAT_TYPE *)mkl_malloc(dimSys*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		#if COMPLEX
		const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, &one, sys_enl->ops[i + model->num_ops], 
			dimSys, sys_enl->ops[1], dimSys, &zero, SSop, dimSys);
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, 1.0, sys_enl->ops[i + model->num_ops], 
			dimSys, sys_enl->ops[1], dimSys, 0.0, SSop, dimSys);
		#endif
		MAT_TYPE* supOp_r = (MAT_TYPE *)mkl_calloc(num_restr_ind*num_restr_ind, sizeof(MAT_TYPE), MEM_DATA_ALIGN);

		kronI_r('R', dimSys, dimEnv, SSop, supOp_r, num_restr_ind, restr_basis_inds);
		mkl_free(SSop);

		transformOps(1, num_restr_ind, 1, psi0_r, &supOp_r);

		#if COMPLEX
		meas->SSs[i] = (*supOp_r).real;
		#else
		meas->SSs[i] = *supOp_r;
		#endif

		mkl_free(supOp_r);
	}

	freeDMRGBlock(sys_enl);
	mkl_free(restr_basis_inds);
	mkl_free(psi0_r);

	return meas;
}

/* Infinite System DMRG Algorithm
   
   L: Maximum length of system
   m: truncation dimension size
*/
void inf_dmrg(const int L, const int m, model_t *model) {
	// TODO: measurement (copy from fin_dmrgR)
	DMRGBlock *sys = createDMRGBlock(model, L);

	while (2*sys->length < L) {
		int currL = sys->length * 2 + 2;
		printf("\nL = %d\n", currL);
		sys = single_step(sys, sys, m, 0, NULL);

		printf("E/L = % .12f\n", sys->energy / currL);
	}

	freeDMRGBlock(sys);
}

/* Finite System DMRG Algorithm
   
   L         : Length of universe
   m_inf     : truncation dimension size for infinite algorithm for building system
   num_sweeps: number of finite system sweeps
   ms        : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
meas_data_t *fin_dmrg(const int L, const int m_inf, const int num_sweeps, int *ms, model_t *model) {

	DMRGBlock **saved_blocksL = (DMRGBlock **)mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
	DMRGBlock **saved_blocksR = (DMRGBlock **)mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);

	DMRGBlock *sys = createDMRGBlock(model, L);

	// Note: saved_blocksL[i] has length i+1
	saved_blocksL[0] = sys;
	saved_blocksR[0] = copyDMRGBlock(sys);
	saved_blocksR[0]->side = 'R';

	// run infinite algorithm to build up system
	while (2*sys->length < L) {
		sys = single_step(sys, sys, m_inf, 0, NULL);

		saved_blocksL[sys->length-1] = sys;
		saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
		saved_blocksR[sys->length-1]->side = 'R';
	}

	// Setup psi0_guess
	MAT_TYPE *psi0_guess = NULL;
	MAT_TYPE **psi0_guessp = &psi0_guess;

	meas_data_t *meas;

	// Finite Sweeps
	DMRGBlock *env;
	for (int i = 0; i < num_sweeps; i++) {
		int m = ms[i];

		while (1) {

			// block for building psi0_guess
			DMRGBlock *env_enl;

			switch (sys->side) {
				case 'L':
					env = saved_blocksR[L - sys->length - 3];
					env_enl = saved_blocksR[L - sys->length - 2];
					break;

				case 'R':
					env = saved_blocksL[L - sys->length - 3];
					env_enl = saved_blocksL[L - sys->length - 2];
					break;
			}

			#if COMPLEX
			const MKL_Complex16 one = {.real=1.0, .imag=0.0};
			const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
			#endif

			if (env_enl->trans == NULL) {
				if (*psi0_guessp != NULL) {
					mkl_free(*psi0_guessp);
					*psi0_guessp = NULL;
				}
			} else if (*psi0_guessp != NULL) {
				// Transform psi0_guess into guess for next iteration
				int d_block_env_enl = env_enl->d_block;
				int d_trans_env_enl = env_enl->d_trans;
				int d_block_sys_enl = sys->d_block*model->d_model;

				MAT_TYPE *temp_guess = reorderKron(*psi0_guessp, d_block_env_enl, sys->d_block, model->d_model);
				
				*psi0_guessp = mkl_realloc(*psi0_guessp, d_block_sys_enl*d_trans_env_enl * sizeof(MAT_TYPE));
				MAT_TYPE *trans_env = env_enl->trans;
				if (env->length > 1) {
					// normal guess
					#if COMPLEX
					cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
								&one, trans_env, d_trans_env_enl, temp_guess, d_block_sys_enl, &zero, *psi0_guessp, d_trans_env_enl);
					#else
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
								1.0, trans_env, d_trans_env_enl, temp_guess, d_block_sys_enl, 0.0, *psi0_guessp, d_trans_env_enl);
					#endif
				} else {
					// transpose of above to account for switching sys and env when the end of the chain is reached
					#if COMPLEX
					cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
								&one, temp_guess, d_block_sys_enl, trans_env, d_trans_env_enl, &zero, *psi0_guessp, d_block_sys_enl);
					#else
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
								1.0, temp_guess, d_block_sys_enl, trans_env, d_trans_env_enl, 0.0, *psi0_guessp, d_block_sys_enl);
					#endif

				}
				mkl_free(temp_guess);
			}

			// Switch sides if at the end of the chain
			if (env->length == 1) {
				DMRGBlock *tempBlock = sys;
				sys = env;
				env = tempBlock;

				if (sys->side == 'L' && i == num_sweeps-1) {
					startMeasBlock(sys);
					printf("Keeping track of measurements...\n");
				}
			}

			// measure and finish run
			if (i == num_sweeps-1 && 2 * sys->length == L-2 && sys->side == 'L') {
				printf("Done with sweep %d/%d\n", num_sweeps, num_sweeps);
				printf("\nTaking measurements...\n");
				meas = meas_step(sys, env, m, 0, psi0_guessp);
				break;
			}

			// printGraphic(sys, env);
			sys = single_step(sys, env, m, 0, psi0_guessp);
			logBlock(sys);

			// Save new block
			switch (sys->side) {
				case 'L':
					if (saved_blocksL[sys->length-1]) { freeDMRGBlock(saved_blocksL[sys->length-1]); }
					saved_blocksL[sys->length-1] = sys;
					break;

				case 'R':
					if (saved_blocksR[sys->length-1]) { freeDMRGBlock(saved_blocksR[sys->length-1]); }
					saved_blocksR[sys->length-1] = sys;
					break;
			}

			// Check if sweep is done
			if (sys->side == 'L' && 2 * sys->length == L) {
				printf("Done with sweep %d/%d\n", i+1, num_sweeps);
				logSweepEnd();
				break;
			}
		}
	}

	if (*psi0_guessp != NULL) { mkl_free(*psi0_guessp); }
	for (int i = 0; i < L-3; i++) {
		if (saved_blocksL[i]) { freeDMRGBlock(saved_blocksL[i]); }
		if (saved_blocksR[i]) { freeDMRGBlock(saved_blocksR[i]); }
	}
	mkl_free(saved_blocksL);
	mkl_free(saved_blocksR);

	return meas;
}

/* Finite System DMRG Algorithm with reflection symmetry in ground state.

   Reflection symmetry means assuming both left and right sides of the system
   are the same so sweeps are only necessary in one direction, halving compute time.
   
   L         : Length of universe
   m_inf     : truncation dimension size for infinite algorithm for building system
   num_sweeps: number of finite system sweeps
   ms        : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
meas_data_t *fin_dmrgR(const int L, const int m_inf, const int num_sweeps, int *ms, model_t *model) {

	DMRGBlock **saved_blocks = (DMRGBlock **)mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);

	DMRGBlock *sys = createDMRGBlock(model, L);

	// Note: saved_blocks[i] has length i+1
	saved_blocks[0] = sys;

	// Run infinite algorithm to build up system
	while (2*sys->length < L) {
		// printGraphic(sys, sys);
		sys = single_step(sys, sys, m_inf, 0, NULL);
		saved_blocks[sys->length-1] = sys;
	}

	// Setup psi0_guess
	MAT_TYPE *psi0_guess = NULL;
	MAT_TYPE **psi0_guessp = &psi0_guess;

	meas_data_t *meas;
	
	// Finite Sweeps
	DMRGBlock *env;
	for (int i = 0; i < num_sweeps; i++) {
		int m = ms[i];

		while (1) {

			env = saved_blocks[L - sys->length - 3];
			DMRGBlock *env_enl = saved_blocks[L - sys->length - 2]; // block for creating psi0_guess

			if (env_enl->trans == NULL) {
				if (*psi0_guessp != NULL) {
					mkl_free(*psi0_guessp);
					*psi0_guessp = NULL;
				}
			} else if (*psi0_guessp != NULL) {
				// Transform psi0_guess into guess for next iteration
				int d_block_env_enl = env_enl->d_block;
				int d_trans_env_enl = env_enl->d_trans;
				int d_block_sys_enl = sys->d_block*model->d_model;

				MAT_TYPE *temp_guess = reorderKron(*psi0_guessp, d_block_env_enl, sys->d_block, model->d_model);
				
				*psi0_guessp = mkl_realloc(*psi0_guessp, d_block_sys_enl*d_trans_env_enl * sizeof(MAT_TYPE));
				MAT_TYPE *trans_env = env_enl->trans;
				#if COMPLEX
				const MKL_Complex16 one = {.real=1.0, .imag=0.0};
				const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
				#endif
				if (env->length > 1) {
					// normal guess
					#if COMPLEX
					cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
								&one, trans_env, d_trans_env_enl, temp_guess, d_block_sys_enl, &zero, *psi0_guessp, d_trans_env_enl);
					#else
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
								1.0, trans_env, d_trans_env_enl, temp_guess, d_block_sys_enl, 0.0, *psi0_guessp, d_trans_env_enl);
					#endif
				} else {
					// transpose of above to account for switching sys and env when the end of the chain is reached
					#if COMPLEX
					cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
								&one, temp_guess, d_block_sys_enl, trans_env, d_trans_env_enl, &zero, *psi0_guessp, d_block_sys_enl);
					#else
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
								1.0, temp_guess, d_block_sys_enl, trans_env, d_trans_env_enl, 0.0, *psi0_guessp, d_block_sys_enl);
					#endif
				}
				mkl_free(temp_guess);
			}

			// Switch sys and env if at the end of the chain
			if (env->length == 1) {
				DMRGBlock *tempBlock = sys;
				sys = env;
				env = tempBlock;

				if (i == num_sweeps-1) {
					startMeasBlock(sys);
					printf("Keeping track of measurements...\n");
				}
			}

			// measure and finish run
			if (i == num_sweeps-1 && 2 * sys->length == L-2) {
				printf("Done with sweep %d/%d\n", num_sweeps, num_sweeps);
				printf("\nTaking measurements...\n");
				meas = meas_step(sys, env, m, 0, psi0_guessp);
				break;
			}

			// printGraphic(sys, env);
			sys = single_step(sys, env, m, 0, psi0_guessp);
			logBlock(sys);

			// Save new block
			if (saved_blocks[sys->length-1]) { freeDMRGBlock(saved_blocks[sys->length-1]); }
			saved_blocks[sys->length-1] = sys;

			// Check if sweep is done
			if (2 * sys->length == L) {
				printf("Done with sweep %d/%d\n", i+1, num_sweeps);
				logSweepEnd();
				break;
			}
		}
	}

	if (*psi0_guessp != NULL) { mkl_free(*psi0_guessp); }
	for (int i = 0; i < L-3; i++) {
		if (saved_blocks[i]) { freeDMRGBlock(saved_blocks[i]); }
	}
	mkl_free(saved_blocks);

	return meas;
}
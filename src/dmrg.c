#include "dmrg.h"
#include "block.h"
#include "model.h"
#include "hamil.h"
#include "sector.h"
#include "meas.h"
#include "linalg.h"
#include "logio.h"
#include "matio.h"
#include "util.h"
#include "uthash.h"
#include <mkl.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

/*  Single DMRG step

	m: truncation dimension size

	step_params:
		target_mz: mz for ground state if symmetry is active

		psi0_guessp: pointer to guess for psi0. Calculated psi0 is returned in this pointer.
		             Set  psi0_guessp = NULL to not use eigenstate guessing and not return eigenstate.
		             Set *psi0_guessp = NULL to not use eigenstate guessing but return eigenstate for future guessing.

	   tau: time advance for TDMRG (set to 0 for normal dmrg)

   returns enlarged system block
*/
DMRGBlock *single_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, dmrg_step_params_t *step_params) {

	int target_mz = step_params->target_mz;
	MAT_TYPE ** psi0_guessp = step_params->psi0_guessp;
	double tau = step_params->tau;

	#ifndef COMPLEX
	if (tau != 0.0) {
		errprintf("Cannot advance time with real matrices. To use tau please compile with -DCOMPLEX flag.\n");
	}
	#endif


	DMRGBlock *sys_enl, *env_enl;
	sector_t *sys_enl_sectors, *env_enl_sectors;
	const model_t *model = sys->model;

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

	// Find ground state
	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	MAT_TYPE *psi0_r = getLowestEStates(sys_enl, env_enl, model, num_restr_ind, restr_basis_inds, 1, psi0_guessp, energies);

	// time tracked state psi
	#if COMPLEX
	MAT_TYPE *psiT_r;
	if (tau != 0) {
		MAT_TYPE *psiTprev_r = restrictVec(sys_enl->psi, num_restr_ind, restr_basis_inds);

		MAT_TYPE *H_int_r = model->H_int_r(model->H_params, sys_enl, env_enl, num_restr_ind, restr_basis_inds);

		MAT_TYPE *expH_r = matExp(num_restr_ind, H_int_r, -1*tau);
		mkl_free(H_int_r);

		const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};

		// time evolve psi(t)
		psiT_r = mkl_malloc(num_restr_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		cblas_zgemv(CblasColMajor, CblasNoTrans, num_restr_ind, num_restr_ind, &one, expH_r, num_restr_ind, 
			psiTprev_r, num_restr_ind, &zero, psiT_r, num_restr_ind);

		mkl_free(expH_r);

		// time evolve psi_0(t)
		complex double *cpsi0_r = (complex double *)psi0_r;
		for (int j=0; j<num_restr_ind; j++) {
			cpsi0_r[j] = cexp(-1*tau*energies[0]) * cpsi0_r[j];
		}
	}
	#endif

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
		// Skip if environment does not have corresponding state
		if (env_enl_mz == NULL) {
			continue;
		}
		int dimSys_sec = sys_enl_mz->num_ind;
		int dimEnv_sec = env_enl_mz->num_ind;
		assert(dimSys_sec * dimEnv_sec == n_sec);

		// target states
		int num_targets = 1;
		#if COMPLEX
		if (tau != 0) {
			num_targets = 2;
		}
		#endif

		MAT_TYPE **targets = mkl_malloc(num_targets * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

		// define target states
		targets[0] = restrictVec(psi0_r, n_sec, sec->inds); // ground state
		// tdmrg also track target state
		#if COMPLEX
		if (tau != 0) {
			targets[1] = restrictVec(psiT_r, n_sec, sec->inds); // tracked state
		}
		#endif

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
		MAT_TYPE *trans_sec = mkl_malloc(dimSys_sec*mm_sec * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		int *isuppz_sec = mkl_malloc(2*dimSys_sec * sizeof(int), MEM_DATA_ALIGN);
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

	mm = (newDimSys < mm) ? newDimSys : mm; // minimize again in case states are dropped because of sectors
	MAT_TYPE *trans = mkl_malloc(dimSys*mm * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

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
		cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, mm, dimEnv, dimSys, &one, trans, dimSys, psi0, dimEnv, &zero, *psi0_guessp, mm);
		#else
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasTrans, mm, dimEnv, dimSys, 1.0 , trans, dimSys, psi0, dimEnv, 0.0  , *psi0_guessp, mm);
		#endif
		mkl_free(psi0);
	}

	mkl_free(restr_basis_inds);
	mkl_free(psi0_r);
	#if COMPLEX
	if (tau != 0) {
		mkl_free(psiT_r);
	}
	#endif
	freeSectors(sys_enl_sectors);
	// Free enlarged environment block
	if (sys != env) {
		freeDMRGBlock(env_enl);
		freeSectors(env_enl_sectors);
	}

	return sys_enl;
}
/*  DMRG step that records measurements.
	Use on the last half sweep to measure operators as the system builds.

	m: truncation dimension size

	returns enlarged system block
*/
meas_data_t *meas_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, dmrg_step_params_t *step_params) {
	// TODO: move meas_step into singl_step with option and measurements into their own functions

	MAT_TYPE ** psi0_guessp = step_params->psi0_guessp;

	DMRGBlock *sys_enl, *env_enl;
	const model_t *model = sys->model;

	sys_enl = enlargeBlock(sys);
	if (sys == env) { // Don't recalculate
		env_enl = sys_enl;
	}
	else {
		env_enl = enlargeBlock(env);
	}

	int dimSys = sys_enl->d_block;
	int dimEnv = env_enl->d_block;
	int dimSup = dimSys * dimEnv;

	// Find ground state
	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	// Find lowest energy states
	MAT_TYPE *psi0 = getLowestEStates(sys_enl, env_enl, model, -1, NULL, 1, psi0_guessp, energies);

	meas_data_t *meas = createMeas(sys_enl->num_ops - model->num_ops);
	meas->energy = energies[0] / (sys_enl->length + env_enl->length);
	mkl_free(energies);

	// Free enlarged environment block
	if (sys != env) {
		freeDMRGBlock(env_enl);
	}

	// Make Measurements
	#if COMPLEX
	const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	#endif

	// <S_i> spins
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* temp = mkl_malloc(dimEnv*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, &one, psi0, dimEnv, sys_enl->ops[i + model->num_ops], dimSys, &zero, temp, dimEnv);
		MKL_Complex16 Szi;
		cblas_zdotc_sub(dimSup, psi0, 1, temp, 1, &Szi);
		meas->Szs[i] = Szi.real;
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, 1.0, psi0, dimEnv, sys_enl->ops[i + model->num_ops], dimSys, 0.0, temp, dimEnv);
		double Szi = cblas_ddot(dimSup, psi0, 1, temp, 1);
		meas->Szs[i] = Szi;
		#endif

		mkl_free(temp);
	}

	// <S_i S_j> correlations
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* SSop = mkl_malloc(dimSys*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		MAT_TYPE* temp = mkl_malloc(dimEnv*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);


		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, &one, sys_enl->ops[i + model->num_ops], dimSys, sys_enl->ops[1], dimSys, &zero, SSop, dimSys);
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, &one, psi0, dimEnv, SSop, dimSys, &zero, temp, dimEnv);
		MKL_Complex16 SSi;
		cblas_zdotc_sub(dimSup, psi0, 1, temp, 1, &SSi);
		meas->SSs[i] = SSi.real;
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, 1.0, sys_enl->ops[i + model->num_ops], dimSys, sys_enl->ops[1], dimSys, 0.0, SSop, dimSys);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, 1.0, psi0, dimEnv, SSop, dimSys, 0.0, temp, dimEnv);
		double SSi = cblas_ddot(dimSup, psi0, 1, temp, 1);
		meas->SSs[i] = SSi;
		#endif

		mkl_free(temp);
		mkl_free(SSop);
	}

	freeDMRGBlock(sys_enl);
	mkl_free(psi0);

	return meas;
}

/*  Infinite System DMRG Algorithm

	params : struct containing the parameters for simulation
		L    : Maximum length of system
		minf : truncation dimension size
*/
meas_data_t *inf_dmrg(sim_params_t *params) {

	const int L    = params->L;
	const int m    = params->minf;
	model_t *model = params->model;

	// step params definition
	dmrg_step_params_t step_params = {};
	step_params.target_mz = 0;
	step_params.psi0_guessp = NULL;

	DMRGBlock *sys = createDMRGBlock(model);
	startMeasBlock(sys);

	while (2*(sys->length-1) < L) {
		int currL = sys->length * 2 + 2;
		printf("\nL = %d\n", currL);
		sys = single_step(sys, sys, m, &step_params);

		printf("E/L = % .12f\n", sys->energy / currL);
	}

	meas_data_t *meas = meas_step(sys, sys, m, &step_params);

	freeDMRGBlock(sys);

	return meas;
}

/*  Finite System DMRG Algorithm
	
	params : struct containing the parameters for simulation
		L      : Length of universe
		m_inf  : truncation dimension size for infinite algorithm for building system
		num_ms : number of finite system sweeps
		ms     : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
meas_data_t *fin_dmrg(sim_params_t *params) {

	const int L          = params->L;
	const int m_inf      = params->minf;
	const int num_sweeps = params->num_ms;
	const int *ms        = params->ms;
	model_t *model       = params->model;

	DMRGBlock **saved_blocksL = mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
	char (*disk_filenamesL)[1024] = mkl_calloc((L-3), sizeof(char[1024]), MEM_DATA_ALIGN);

	DMRGBlock **saved_blocksR;
	char (*disk_filenamesR)[1024];
	if (params->reflection) {
		saved_blocksR = saved_blocksL;
		disk_filenamesR = disk_filenamesL;
	} else {
		saved_blocksR = mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
		disk_filenamesR = mkl_calloc((L-3), sizeof(char[1024]), MEM_DATA_ALIGN);
	}


	// NOTE: These macros are unsafe (do not use something like x++ as the parameter)
	// Macro to save block at index to disk
	#define SAVE_SIDE_BLOCK_TO_DISK(ind, side) if (params->save_blocks) { \
		char *refside = params->reflection ? "L" : #side; \
		sprintf(disk_filenames##side[ind], "%s/%s%05d.blk", params->block_dir, refside, ind); \
		saveBlock(disk_filenames##side[ind], saved_blocks##side[ind]); }

	// Macro to read block from disk
	// Only reads if block not already loaded in RAM
	#define LOAD_SIDE_BLOCK_FROM_DISK(ind, side) if (params->save_blocks && disk_filenames##side[ind][0] != '\0') { \
		readBlock(disk_filenames##side[ind], saved_blocks##side[ind]); \
		disk_filenames##side[ind][0] = '\0'; }


	// step params definition
	dmrg_step_params_t step_params = {};
	step_params.target_mz = 0;
	step_params.psi0_guessp = NULL;

	DMRGBlock *sys;
	DMRGBlock *env;

	if (params->continue_run) { // use saved blocks

		printf("Continuing run using saved blocks at '%s' ...\n", params->block_dir);

		for (int i=0; i<L-3; i++) {
			sprintf(disk_filenamesL[i], "%s/L%05d.blk", params->block_dir, i);
			saved_blocksL[i] = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
			saved_blocksL[i]->model = model;

			if (!params->reflection) {
				sprintf(disk_filenamesR[i], "%s/R%05d.blk", params->block_dir, i);
				saved_blocksR[i] = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
				saved_blocksR[i]->model = model;
			}
		}

		int ret = readBlock(disk_filenamesL[L-4], saved_blocksL[L-4]);
		disk_filenamesL[L-4][0] = '\0';
		if (ret != 0) {
			errprintf("Could not load dmrg block.\n");
			exit(1);
		}
		sys = saved_blocksL[L-4];

	} else { // build system normally

		sys = createDMRGBlock(model);

		// Note: saved_blocksL[i] has length i+1
		saved_blocksL[0] = sys;
		if (!params->reflection) {
			saved_blocksR[0] = copyDMRGBlock(sys);
			saved_blocksR[0]->side = 'R';
		}

		// run infinite algorithm to build up system
		while (2*sys->length < L) {
			#ifndef NDEBUG
			printGraphic(sys, sys);
			#endif
			sys = single_step(sys, sys, m_inf, &step_params);

			saved_blocksL[sys->length-1] = sys;
			if (!params->reflection) {
				saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
				saved_blocksR[sys->length-1]->side = 'R';
			}

			// write old block to disk
			int sys_old_index = sys->length-2;
			SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, L);
			if (!params->reflection) {
				SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, R);
			}
		}
	}



	// Setup psi0_guess
	MAT_TYPE *psi0_guess = NULL;
	MAT_TYPE **psi0_guessp = &psi0_guess;
	step_params.psi0_guessp = &psi0_guess;

	meas_data_t *meas;

	// Finite Sweeps
	for (int i = 0; i < num_sweeps; i++) {
		int m = ms[i];

		while (1) {

			int env_index = L - sys->length - 3;
			int env_enl_index = L - sys->length - 2;
			DMRGBlock *env_enl;
			switch (sys->side) {
				case 'L':
					env = saved_blocksR[env_index];
					LOAD_SIDE_BLOCK_FROM_DISK(env_index, R);
					env_enl = saved_blocksR[env_enl_index];
					break;

				case 'R':
					env = saved_blocksL[env_index];
					LOAD_SIDE_BLOCK_FROM_DISK(env_index, L);
					env_enl = saved_blocksL[env_enl_index];
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
				// Load env_enl block for converting psi0_guess to the right basis
				switch (sys->side) {
					case 'L':
						LOAD_SIDE_BLOCK_FROM_DISK(env_enl_index, R);
						break;
					case 'R':
						LOAD_SIDE_BLOCK_FROM_DISK(env_enl_index, L);
						break;
				}

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

				// Save enl_block back to disk
				switch (sys->side) {
					case 'L':
						SAVE_SIDE_BLOCK_TO_DISK(env_enl_index, R);
						break;

					case 'R':
						SAVE_SIDE_BLOCK_TO_DISK(env_enl_index, L);
						break;
				}
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
				meas = meas_step(sys, env, m, &step_params);
				break;
			}
			
			#ifndef NDEBUG
			printGraphic(sys, env);
			#endif
			sys = single_step(sys, env, m, &step_params);
			logBlock(sys);

			// Save new block
			int sys_index = sys->length-1;
			switch (sys->side) {
				case 'L':
					// if block saved to disk only free pointer. Otherwise free matrices too.
					if (disk_filenamesL[sys_index][0] != '\0') {
						mkl_free(saved_blocksL[sys_index]);
						disk_filenamesL[sys_index][0] = '\0';
					} else if (saved_blocksL[sys_index]) {
						freeDMRGBlock(saved_blocksL[sys_index]);
					}
					saved_blocksL[sys_index] = sys;

					// write old block when not on measuring sweep
					if (sys->meas != 'M') {
						int sys_old_index = sys->length-2;
						SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, L);
					} 
					break;

				case 'R':
					// if block saved to disk only free pointer. Otherwise free matrices too.
					if (disk_filenamesR[sys_index][0] != '\0') {
						mkl_free(saved_blocksR[sys_index]);
						disk_filenamesR[sys_index][0] = '\0';
					} else if (saved_blocksR[sys_index]) {
						freeDMRGBlock(saved_blocksR[sys_index]);
					}
					saved_blocksR[sys_index] = sys;

					// write old block when not on measuring sweep
					if (sys->meas != 'M') {
						int sys_old_index = sys->length-2;
						SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, R);
					} 
					break;
			}

			// Check if sweep is done
			if (sys->side == 'L' && 2 * sys->length == L) {
				printf("Done with sweep %d/%d with m=%d.\n", i+1, num_sweeps, m);
				logSweepEnd();
				break;
			}
		}
	}

	if (*psi0_guessp != NULL) { mkl_free(*psi0_guessp); }
	for (int i = 0; i < L-3; i++) {
		if (disk_filenamesL[i][0] != '\0') { mkl_free(saved_blocksL[i]); }
		else if (saved_blocksL[i]) { freeDMRGBlock(saved_blocksL[i]); }

		if (!params->reflection) {
			if (disk_filenamesR[i][0] != '\0') { mkl_free(saved_blocksR[i]); }
			else if (saved_blocksR[i]) { freeDMRGBlock(saved_blocksR[i]); }
		}
	}
	mkl_free(saved_blocksL);
	mkl_free(disk_filenamesL);
	if (!params->reflection) {
		mkl_free(saved_blocksR);
		mkl_free(disk_filenamesR);
	}

	return meas;
}

/*  Finite System DMRG Algorithm with reflection symmetry in ground state.

	Reflection symmetry means assuming both left and right sides of the system
	are the same so sweeps are only necessary in one direction, halving compute time.

	params : struct containing the parameters for simulation
		L      : Length of universe
		m_inf  : truncation dimension size for infinite algorithm for building system
		num_ms : number of finite system sweeps
		ms     : (size num_ms) list of truncation sizes for the finite sweeps

*/
meas_data_t *fin_dmrgR(sim_params_t *params) {

	const int L          = params->L;
	const int m_inf      = params->minf;
	const int num_sweeps = params->num_ms;
	const int *ms        = params->ms;
	model_t *model       = params->model;

	DMRGBlock **saved_blocks = mkl_calloc((L-3), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
	// filenames for blocks saved to disk. When block i is loaded in RAM, disk_filenames[i] is set to '\0'
	char (*disk_filenames)[1024] = mkl_calloc((L-3), sizeof(char[1024]), MEM_DATA_ALIGN);


	// NOTE: These macros are unsafe (do not use something like x++ as the parameter)
	// Macro to save block at index to disk
	#define SAVE_BLOCK_TO_DISK(ind) if (params->save_blocks) { \
		sprintf(disk_filenames[ind], "%s/%05d.blk", params->block_dir, ind); \
		saveBlock(disk_filenames[ind], saved_blocks[ind]); }

	// Macro to load block at index from disk
	// Only reads if block not already laoded in RAM
	#define LOAD_BLOCK_FROM_DISK(ind) if (params->save_blocks && disk_filenames[ind][0] != '\0') { \
		readBlock(disk_filenames[ind], saved_blocks[ind]); \
		disk_filenames[ind][0] = '\0'; }


	// step params definition
	dmrg_step_params_t step_params = {};
	step_params.target_mz = 0;
	step_params.psi0_guessp = NULL;

	DMRGBlock *sys;
	DMRGBlock *env;

	if (params->continue_run) { // use saved blocks

		printf("Continuing run using saved blocks at '%s' ...\n", params->block_dir);

		for (int i=0; i<L-3; i++) {
			sprintf(disk_filenames[i], "%s/%05d.blk", params->block_dir, i);
			saved_blocks[i] = mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
			saved_blocks[i]->model = model;
		}

		int ret = readBlock(disk_filenames[L-4], saved_blocks[L-4]);
		disk_filenames[L-4][0] = '\0';
		if (ret != 0) {
			errprintf("Could not load dmrg block.\n");
			exit(1);
		}
		sys = saved_blocks[L-4];

	} else { // build system normally

		sys = createDMRGBlock(model);

		// Note: saved_blocks[i] has length i+1
		saved_blocks[0] = sys;

		// Run infinite algorithm to build up system
		while (2*sys->length < L) {
			#ifndef NDEBUG
			printGraphic(sys, sys);
			#endif
			sys = single_step(sys, sys, m_inf, &step_params);
			saved_blocks[sys->length-1] = sys;

			// write old block to disk
			int sys_old_index = sys->length-2;
			SAVE_BLOCK_TO_DISK(sys_old_index);
		}
	}


	// Setup psi0_guess
	MAT_TYPE *psi0_guess = NULL;
	MAT_TYPE **psi0_guessp = &psi0_guess;
	step_params.psi0_guessp = &psi0_guess;

	meas_data_t *meas;
	
	// Finite Sweeps
	for (int i = 0; i < num_sweeps; i++) {
		int m = ms[i];

		while (1) {

			int env_index = L - sys->length - 3;
			env = saved_blocks[env_index];
			LOAD_BLOCK_FROM_DISK(env_index);

			int env_enl_index = L - sys->length - 2;
			DMRGBlock *env_enl = saved_blocks[env_enl_index];

			if (env_enl->trans == NULL) {
				if (*psi0_guessp != NULL) {
					mkl_free(*psi0_guessp);
					*psi0_guessp = NULL;
				}
			} else if (*psi0_guessp != NULL) {
				// Load env_enl block for converting psi0_guess to the right basis
				LOAD_BLOCK_FROM_DISK(env_enl_index);

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

				// Save enl_block back to disk
				SAVE_BLOCK_TO_DISK(env_enl_index);
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
				printf("Done with sweep %d/%d with m=%d.\n", num_sweeps, num_sweeps, m);
				printf("\nTaking measurements...\n");
				meas = meas_step(sys, env, m, &step_params);
				break;
			}

			#ifndef NDEBUG
			printGraphic(sys, env);
			#endif
			sys = single_step(sys, env, m, &step_params);
			logBlock(sys);

			// Save new block
			int sys_index = sys->length-1;
			// if block saved to disk only free pointer. Otherwise free matrices too.
			if (disk_filenames[sys_index][0] != '\0') {
				mkl_free(saved_blocks[sys_index]);
				disk_filenames[sys_index][0] = '\0';
			} else if (saved_blocks[sys_index]) {
				freeDMRGBlock(saved_blocks[sys_index]);
			}
			saved_blocks[sys_index] = sys;

			// write old block when not on measuring sweep
			if (sys->meas != 'M') {
				int sys_old_index = sys->length-2;
				SAVE_BLOCK_TO_DISK(sys_old_index);
			} 

			// Check if sweep is done
			if (2 * sys->length == L) {
				printf("Done with sweep %d/%d with m=%d.\n", i+1, num_sweeps, m);
				logSweepEnd();
				break;
			}
		}
	}

	if (*psi0_guessp != NULL) { mkl_free(*psi0_guessp); }
	for (int i = 0; i < L-3; i++) {
		if (disk_filenames[i][0] != '\0') { mkl_free(saved_blocks[i]); }
		else if (saved_blocks[i]) { freeDMRGBlock(saved_blocks[i]); }
	}
	mkl_free(saved_blocks);
	mkl_free(disk_filenames);

	return meas;
}
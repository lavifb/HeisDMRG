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
	MAT_TYPE **psi_tp = step_params->psi_tp;

	if (tau != 0.0 && (psi_tp == NULL || *psi_tp == NULL)) {
		errprintf("No time dep state psi_t provided.\nContinuing without time evolution...");
		tau = 0;
	}

	#ifndef COMPLEX
	if (tau != 0.0) {
		errprintf("Cannot advance time with real matrices. To use tau please compile with -DCOMPLEX flag.\nContinuing without time evolution...");
		tau = 0;
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

	// sup_sectors stores sectors for superblock
	sector_t *sup_sectors;

	if (step_params->abelianSectorize) {
		// Get restricted basis and set restr_basis_inds and num_restr_ind
		sup_sectors = getRestrictedBasis(sys_enl_sectors, env_enl_sectors, target_mz, dimEnv, &num_restr_ind, restr_basis_inds);
	} else {
		sup_sectors = NULL;
		// placeholder that isn't really used
		sector_t *sec = createSector(0);
		for (int i=0; i<dimSup; i++) {
			sectorPush(sec, i);
			restr_basis_inds[i] = i;
		}
		num_restr_ind = dimSup;
		HASH_ADD_INT(sup_sectors, id, sec);
	}

	// Find ground state
	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	// Find lowest energy states
	MAT_TYPE *psi0 = getLowestEStates(sys_enl, env_enl, model, 1, psi0_guessp, energies);
	MAT_TYPE *psi0_r = restrictVec(psi0, num_restr_ind, restr_basis_inds);

	// time tracked state psi
	MAT_TYPE *psiT_r;
	if (tau != 0) {
		MAT_TYPE *psiTprev_r = restrictVec(*psi_tp, num_restr_ind, restr_basis_inds);

		MAT_TYPE *H_int_r = model->H_int_r(model, sys_enl, env_enl, num_restr_ind, restr_basis_inds);

		MAT_TYPE *expH_r = matExp(num_restr_ind, H_int_r, -.5*I*tau);
		mkl_free(H_int_r);

		const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
		const MKL_Complex16 zero = {.real=0.0, .imag=0.0};

		// time evolve psi(t)
		psiT_r = mkl_malloc(num_restr_ind * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		cblas_zgemv(CblasColMajor, CblasNoTrans, num_restr_ind, num_restr_ind, &one, expH_r, num_restr_ind, 
			psiTprev_r, 1, &zero, psiT_r, 1);

		mkl_free(psiTprev_r);
		mkl_free(expH_r);

		// time evolve psi_0(t)
        complex double *cpsi0_r = (complex double *)psi0_r;
        complex double time_rot = cexp(-.5*I*tau*energies[0]);
        for (int j=0; j<num_restr_ind; j++) {
            cpsi0_r[j] = time_rot * cpsi0_r[j];
        }
	}

	if (step_params->measure) {
		meas_data_t *meas = createMeas(sys_enl->num_ops - model->num_ops);
		meas->energy = energies[0] / (sys_enl->length + env_enl->length);

		measureSzs(sys_enl, dimEnv, psi0, model->num_ops, meas);
		measureSSs(sys_enl, dimEnv, psi0, model->num_ops, meas);

		step_params->meas = meas;
	}
	mkl_free(psi0);

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

		int mz;
		int n_sec = sec->num_ind;
		int dimSys_sec = dimSys;
		int dimEnv_sec = dimEnv;
		sector_t *sys_enl_mz, *env_enl_mz;


		if (step_params->abelianSectorize) {
			mz = sec->id;
			// printf("mz = %d\n", mz);
			int env_mz = target_mz - mz;

			HASH_FIND_INT(sys_enl_sectors, &mz    , sys_enl_mz);
			HASH_FIND_INT(env_enl_sectors, &env_mz, env_enl_mz);
			assert(sys_enl_mz != NULL);
			// Skip if environment does not have corresponding state
			if (env_enl_mz == NULL) {
				continue;
			}
			dimSys_sec = sys_enl_mz->num_ind;
			dimEnv_sec = env_enl_mz->num_ind;
			assert(dimSys_sec * dimEnv_sec == n_sec);
		}

		// target states
		int num_targets = 1;
		if (tau != 0) {
			num_targets = 2;
		}

		MAT_TYPE **targets = mkl_malloc(num_targets * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
		// define target states
		targets[0] = restrictVec(psi0_r, n_sec, sec->inds); // ground state
		// tdmrg also track target state
		if (tau != 0) {
			targets[1] = restrictVec(psiT_r, n_sec, sec->inds); // tracked state
		}


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
				if (step_params->abelianSectorize) {
					trans_full[lamb_i*dimSys + sys_enl_mz->inds[j]] = trans_sec[i*dimSys_sec + j];
				} else {
					trans_full[lamb_i*dimSys + j] = trans_sec[i*dimSys_sec + j];
				}
			}
			if (step_params->abelianSectorize) {
				// keep track of mzs for the enlarged block
				sys_mzs_full[lamb_i] = mz;
			} else {
				sys_mzs_full[lamb_i] = sys_enl->mzs[lamb_i];
			}
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
			*psi0_guessp = mkl_malloc(dimSup * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		} else {
			// Check overlap of guess and calculated eigenstate
			#define PRINT_OVERLAP
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

				if (overlap < .9) {
					printf("Guess is bad!!\n");
					print_matrix("psi0_guess", dimEnv, dimSys, *psi0_guessp, dimEnv);
					print_matrix("psi0"      , dimEnv, dimSys, psi0        , dimEnv);

					printf("\ndimSys = %3d  dimEnv = %3d  dimSup = %3d\n", dimSys, dimEnv, dimSup);
					// exit(1);
				}
			#endif

			*psi0_guessp = mkl_realloc(*psi0_guessp, dimSup * sizeof(MAT_TYPE));
		}

		memcpy(*psi0_guessp, psi0, dimSup * sizeof(MAT_TYPE));
		mkl_free(psi0);
	}

	if (tau != 0.0) {
		MAT_TYPE *psi_t = unrestrictVec(dimSup, psiT_r, num_restr_ind, restr_basis_inds);
		mkl_free(psiT_r);
		*psi_tp = mkl_realloc(*psi_tp, dimSup * sizeof(MAT_TYPE));

		memcpy(*psi_tp, psi_t, dimSup * sizeof(MAT_TYPE));
		mkl_free(psi_t);
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

/*  Davidson transform to change basis for state psi for next iteration

    psip: pointer to state psi
*/
void DavidsonTransform(DMRGBlock *sys, DMRGBlock *env_enl, MAT_TYPE **psip) {

	// TODO: fix transform in 1 case when using reflection symmetry

	int d_model = sys->model->d_model;
	int d_trans_sys = sys->d_trans;
	int d_block_env_enl = env_enl->d_block;
	int d_trans_env_enl = env_enl->d_trans;
	int d_block_sys_enl = sys->d_block*d_model;

	MAT_TYPE *psi_temp = mkl_malloc(d_block_env_enl*d_model*d_trans_sys  * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	#if COMPLEX
	const MKL_Complex16 one = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, sys->d_block, d_block_env_enl*d_model, d_trans_sys, &one, sys->trans, d_trans_sys, *psip, d_block_env_enl*d_model, &zero, psi_temp, sys->d_block);
	#else
	cblas_dgemm(CblasColMajor, CblasConjTrans, CblasTrans, sys->d_block, d_block_env_enl*d_model, d_trans_sys, 1.0 , sys->trans, d_trans_sys, *psip, d_block_env_enl*d_model, 0.0  , psi_temp, sys->d_block);
	#endif

	MAT_TYPE *psi_temp2 = reorderKron(psi_temp, d_block_env_enl, sys->d_block, d_model);
	
	*psip = mkl_realloc(*psip, d_block_sys_enl*d_trans_env_enl * sizeof(MAT_TYPE));
	if (env_enl->length > 2) {
		// normal guess
		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
					&one, env_enl->trans, d_trans_env_enl, psi_temp2, d_block_sys_enl, &zero, *psip, d_trans_env_enl);
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_trans_env_enl, d_block_sys_enl, d_block_env_enl, 
					1.0, env_enl->trans, d_trans_env_enl, psi_temp2, d_block_sys_enl, 0.0, *psip, d_trans_env_enl);
		#endif
	} else {
		// transpose of above to account for switching sys and env when the end of the chain is reached
		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
					&one, psi_temp2, d_block_sys_enl, env_enl->trans, d_trans_env_enl, &zero, *psip, d_block_sys_enl);
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, d_block_sys_enl, d_trans_env_enl, d_block_env_enl, 
					1.0, psi_temp2, d_block_sys_enl, env_enl->trans, d_trans_env_enl, 0.0, *psip, d_block_sys_enl);
		#endif

	}
	mkl_free(psi_temp);
	mkl_free(psi_temp2);
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
	step_params.abelianSectorize = 1;
	step_params.target_mz = 0;
	step_params.psi0_guessp = NULL;

	DMRGBlock *sys = createDMRGBlock(model);
	startMeasBlock(sys);

	while (2*(sys->length-1) < L) {
		int currL = sys->length * 2 + 2;
		printf("\nL = %d\n", currL);
		DMRGBlock *new_sys = single_step(sys, sys, m, &step_params);
		
		freeDMRGBlock(sys);
		sys = new_sys;

		printf("E/L = % .12f\n", sys->energy / currL);
	}

	step_params.measure = 1; // take measurements
	DMRGBlock *sys_extra = single_step(sys, sys, m, &step_params);
	freeDMRGBlock(sys_extra);

	freeDMRGBlock(sys);

	return step_params.meas;
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
	const int num_sweeps = params->num_ms + params->num_ts;
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
	step_params.abelianSectorize = 1;
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
		
		int m;

		if (i < params->num_ms) {
			m = ms[i];
		} else if (i == params->num_ms) {
			// prepare TDMRG sweeps
			m = ms[params->num_ms-1];

			// create time tracked state psi_t
			int env_index = L - sys->length - 3;
			env = saved_blocksR[env_index];
			LOAD_SIDE_BLOCK_FROM_DISK(env_index, R);

			int dimSys = sys->d_block * model->d_model;
			int dimEnv = env->d_block * model->d_model;
			int dimSup = dimSys * dimEnv;

			MAT_TYPE *psi_t = mkl_malloc(dimSup * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

			// Create matrix to apply Sp to central spin
			MAT_TYPE *Sp_sys = mkl_calloc(dimSys*dimSys, sizeof(MAT_TYPE), MEM_DATA_ALIGN);
			kronI('L', sys->d_block, model->d_model, model->Sp, Sp_sys);

			// Apply Sp to central spin
			const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
			const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
			cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, &one, *step_params.psi0_guessp, dimEnv, Sp_sys, dimSys, &one, psi_t, dimEnv);
			
			step_params.psi_tp = &psi_t;
			step_params.tau = params->dtau;
		} else {
			m = ms[params->num_ms-1];
		}

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

			if (env_enl->trans == NULL || sys->trans == NULL) {
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

				// convert psi0_guess into new basis
				DavidsonTransform(sys, env_enl, psi0_guessp);
				// convert psi_t during tdmrg sweeps
				if (i >= params->num_ms) {
					DavidsonTransform(sys, env_enl, step_params.psi_tp);
				}

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
			if (sys->meas == 'M' && sys->length == env->length) {
				printf("Done with sweep %d/%d\n", num_sweeps, num_sweeps);
				printf("\nTaking measurements...\n");
				step_params.measure = 1; // take measurements
				sys = single_step(sys, env, m, &step_params);
				meas = step_params.meas;
				if (params->num_ts > 0) {
					char measFilename[1024];
					sprintf(measFilename, "%smeas_t%f.dat", params->block_dir, params->dtau*(i-params->num_ms+1));
					outputMeasData(measFilename, step_params.meas);
				}
			} else { // normal step
				#ifndef NDEBUG
				printGraphic(sys, env);
				#endif
				sys = single_step(sys, env, m, &step_params);
			}
			
			logBlock(sys);

			// Save new block
			int sys_index = sys->length-1;
			int sys_old_index = sys->length-2;
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
					dropMeasurements(saved_blocksL[sys_old_index]);
					SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, L);
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
					dropMeasurements(saved_blocksR[sys_old_index]);
					SAVE_SIDE_BLOCK_TO_DISK(sys_old_index, R);
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

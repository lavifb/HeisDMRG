#include "dmrg.h"
#include "block.h"
#include "linalg.h"
#include "uthash.h"
#include <mkl.h>
#include <mkl_scalapack.h>
#include <assert.h>
// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>

/* Print nice graphic of the system and environment
*/
void printGraphic(DMRGBlock *sys, DMRGBlock *env) {

	char *sys_g = (char *)malloc((sys->length+1) * sizeof(char));
	char *env_g = (char *)malloc((env->length+1) * sizeof(char));

	memset(sys_g, '=', sys->length);
	memset(env_g, '-', env->length);
	sys_g[sys->length] = '\0';
	env_g[env->length] = '\0';

	if (sys->side == 'L') {
		printf("%s**%s\n", sys_g, env_g);
	} else {
		printf("%s**%s\n", env_g, sys_g);
	}

	free(sys_g);
	free(env_g);
}

/* Single DMRG step
   
   m: truncation dimension size

   returns enlarged system block
*/
DMRGBlock *single_step(DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz) {

	DMRGBlock *sys_enl, *env_enl;
	sector_t *sys_enl_sectors, *env_enl_sectors;
	ModelParams *model = sys->model;

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

	double *Isys = identity(dimSys);
	double *Ienv = identity(dimEnv);

	// Superblock Hamiltonian
	double *Hs = HeisenH_int(model->J, model->Jz, dimSys, dimEnv, 
					sys_enl->ops[1], sys_enl->ops[2], env_enl->ops[1], env_enl->ops[2]);
	kron(1.0, dimSys, dimEnv, sys_enl->ops[0], Ienv, Hs);
	kron(1.0, dimSys, dimEnv, Isys, env_enl->ops[0], Hs);

	mkl_free(Isys);
	mkl_free(Ienv);

	// Create sectors to treat seperately
	sector_t *sup_sectors = NULL;

	// indexes used for restricting Hs
	int num_restr_ind = 0;
	int *restr_basis_inds = (int *)mkl_malloc(dimSup * sizeof(int), MEM_DATA_ALIGN);

	sector_t *sys_enl_sec;

	// loop over sys_enl_sectors
	for(sys_enl_sec=sys_enl_sectors; sys_enl_sec != NULL; sys_enl_sec=sys_enl_sec->hh.next) {

		int sys_mz = sys_enl_sec->id;

		sector_t *sup_sec = (sector_t *)mkl_malloc(sizeof(sector_t), MEM_DATA_ALIGN);
		sup_sec->id = sys_mz;
		sup_sec->num_ind = 0;
		HASH_ADD_INT(sup_sectors, id, sup_sec);

		int *sys_inds = sys_enl_sec->inds;
		int env_mz = target_mz - sys_mz;

		sector_t *env_enl_sec;
		HASH_FIND_INT(env_enl_sectors, &env_mz, env_enl_sec);
		if (env_enl_sec != NULL) {
			int i, j;
			for (i = 0; i < sys_enl_sec->num_ind; i++) {
				for (j = 0; j < env_enl_sec->num_ind; j++) {
					// save restriced index and save into sup_sectors
					sup_sec->inds[sup_sec->num_ind] = num_restr_ind;
					sup_sec->num_ind++;
					restr_basis_inds[num_restr_ind] = i*dimEnv + j;
				}
			}
		}
	}

	double *Hs_r = restrictOp(dimSup, Hs, num_restr_ind, restr_basis_inds);
	mkl_free(Hs);
	mkl_free(restr_basis_inds);

	__assume_aligned(Hs_r, MEM_DATA_ALIGN);

	// Find ground state
	double *psi0_r = (double *)mkl_malloc(num_restr_ind * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(psi0_r, MEM_DATA_ALIGN);
	int info;
	int num_es_found;
	double *energies = (double *)mkl_malloc(num_restr_ind * sizeof(double), MEM_DATA_ALIGN);;
	int *ifail = (int *)mkl_malloc(num_restr_ind * sizeof(int), MEM_DATA_ALIGN);;
	__assume_aligned(ifail, MEM_DATA_ALIGN);
	info = LAPACKE_dsyevx(LAPACK_COL_MAJOR, 'V', 'I', 'U', num_restr_ind, Hs_r, num_restr_ind, 
			0.0, 0.0, 1, 1, 0.0, &num_es_found, energies, psi0_r, num_restr_ind, ifail);
	if (info > 0) {
		printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
		exit(1);
	}
	mkl_free(ifail);
	mkl_free(Hs_r);

	double energy = energies[0]; // record ground state energy
	printf("E/L = %6.10f\n", energy / (sys_enl->length + env_enl->length));
	mkl_free(energies);

	// Transformation Matrix
	int mm = (dimSys < m) ? dimSys : m; // use min(dimSys, m) 
	double *trans_full = (double *)mkl_calloc(dimSys*dimSys, sizeof(double), MEM_DATA_ALIGN);

	// Eigenvalues
	int lamb_i = 0;
	double *lambs = (double *)mkl_malloc(dimSys * sizeof(double), MEM_DATA_ALIGN);

	sector_t *sec;
	for(sec=sup_sectors; sec != NULL; sec=sec->hh.next) {
		int mz = sec->id;
		int env_mz = target_mz - mz;
		int n_sec = sec->num_ind;

		double *psi0_sec = restrictVec(num_restr_ind, psi0_r, n_sec, sec->inds);

		sector_t *sys_enl_mz, *env_enl_mz;
		HASH_FIND_INT(sys_enl_sectors, &mz    , sys_enl_mz);
		HASH_FIND_INT(env_enl_sectors, &env_mz, env_enl_mz);
		int dimSys_sec = sys_enl_mz->num_ind;
		int dimEnv_sec = env_enl_mz->num_ind;

		// psi0 needs to be arranged as a dimSys * dimEnv to trace out env
		// Put sys_basis on rows and env_basis on the cols by taking transpose
		// To not take transpose twice, take conj and take conjTrans on left side of dgemm bellow
		mkl_dimatcopy('C', 'R', dimEnv_sec, dimSys_sec, 1.0, psi0_sec, dimEnv_sec, dimEnv_sec);

		// Density matrix rho
		double *rho_sec = (double *)mkl_malloc(dimSys_sec*dimSys_sec * sizeof(double), MEM_DATA_ALIGN);
		__assume_aligned(rho_sec, MEM_DATA_ALIGN);
		// Trace out Environment to make rho (Note transpose structure as described above)
		cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, dimSys_sec, dimSys_sec, dimEnv_sec, 
					1.0, psi0_sec, dimEnv_sec, psi0_sec, dimEnv_sec, 0.0, rho_sec, dimSys_sec);
		mkl_free(psi0_sec);

		// TODO: diagonalize rho_sec and add to list of eignvalues
		//			Then, sort and pick out first mm eigenvectors to make trans.

		// double *lambs_sec = (double *)mkl_malloc(dimSys_sec * sizeof(double), MEM_DATA_ALIGN);
		// __assume_aligned(lambs_sec, MEM_DATA_ALIGN);

		int mm_sec = (dimSys_sec < mm) ? dimSys_sec : mm;
		double *trans_sec = (double *)mkl_malloc(dimSys_sec*mm_sec * sizeof(double), MEM_DATA_ALIGN);
		__assume_aligned(trans_sec, MEM_DATA_ALIGN);
		int *ifail_sec = (int *)mkl_malloc(dimSys_sec * sizeof(int), MEM_DATA_ALIGN);
		info = LAPACKE_dsyevx(LAPACK_COL_MAJOR, 'V', 'I', 'U', dimSys_sec, rho_sec, dimSys_sec, 0.0, 0.0,
				dimSys_sec-mm_sec+1, dimSys_sec, 0.0, &num_es_found, lambs+lamb_i, trans_sec, dimSys_sec, ifail_sec);
		if (info > 0) {
			printf("Failed to find eigenvalues of density matrix\n");
			exit(1);
		}
		mkl_free(rho_sec);
		mkl_free(ifail_sec);

		// copy trans_sec into trans using the proper basis
		int i, j;
		for (i = 0; i < mm_sec; i++) {
			for (j = 0; j < dimSys_sec; j++) {
				// copy value using proper index basis
				trans_full[lamb_i*dimSys + sys_enl_sec->inds[j]] = trans_sec[i*dimSys_sec + j];
			}
			lamb_i++;
		}
		
		// mkl_free(lambs_sec);
		mkl_free(trans_sec);
	}

	double *trans = (double *)mkl_calloc(dimSys*mm, sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(trans, MEM_DATA_ALIGN);

	char SORT_DESCENDING = 'D';

	int *sorted_inds = dsort2(dimSys, lambs);
	if (info < 0) {
		printf("Failed to sort eigenvalues\n");
		exit(1);
	}

	// copy to trans in right order
	int i;
	for (i = 0; i < mm; i++) {
		memcpy(&trans[i*dimSys], &trans_full[sorted_inds[i]*dimSys], dimSys * sizeof(double));
	}

	mkl_free(trans_full);
	
	double truncation_err = 1;
	for (i = 0; i < mm; i++) {
		truncation_err -= lambs[i];
	}
	printf("Truncation Error: %.10e\n", truncation_err);
	mkl_free(lambs);
	mkl_free(sorted_inds);

	// Transform operators into new basis
	transformOps(sys_enl->num_ops, dimSys, mm, trans, sys_enl->ops);
	sys_enl->d_block = mm; // set block basis size to transformed value
	mkl_free(trans);

	// Free enlarged environment block
	if (sys_enl != env_enl) {
		freeDMRGBlock(env_enl);
	}

	return sys_enl;
}

/* Infinite System DMRG Algorithm
   
   L: Maximum length of system
   m: truncation dimension size
*/
void inf_dmrg(const int L, const int m, ModelParams *model) {

	DMRGBlock *sys = createDMRGBlock(model);

	while (2*sys->length < L) {
		printf("\nL = %d\n", sys->length * 2 + 2);
		DMRGBlock *newSys = single_step(sys, sys, m, 0);
		freeDMRGBlock(sys);
		sys = newSys;
	}

	freeDMRGBlock(sys);
}

/* Finite System DMRG Algorithm
   
   L         : Length of universe
   m_inf     : truncation dimension size for infinite algorithm for building system
   num_sweeps: number of finite system sweeps
   ms        : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
void fin_dmrg(const int L, const int m_inf, const int num_sweeps, int *ms, ModelParams *model) {
	assert(L%2 == 0);

	DMRGBlock **saved_blocksL = (DMRGBlock **)mkl_calloc((L-4), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
	DMRGBlock **saved_blocksR = (DMRGBlock **)mkl_calloc((L-4), sizeof(DMRGBlock *), MEM_DATA_ALIGN);

	DMRGBlock *sys = createDMRGBlock(model);

	// Note: saved_blocksL[i] has length i+1
	saved_blocksL[0] = copyDMRGBlock(sys);
	saved_blocksR[0] = copyDMRGBlock(sys);
	saved_blocksR[0]->side = 'R';

	// run infinite algorithm to build up system
	while (2*sys->length < L) {
		printGraphic(sys, sys);
		DMRGBlock *newSys = single_step(sys, sys, m_inf, 0);
		freeDMRGBlock(sys);
		sys = newSys;

		saved_blocksL[sys->length-1] = copyDMRGBlock(sys);
		saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
		saved_blocksR[sys->length-1]->side = 'R';
	}

	// Finite Sweeps
	DMRGBlock *env = copyDMRGBlock(sys);
	int i;
	for (i = 0; i < num_sweeps; i++) {
		int m = ms[i];

		while (1) {
			freeDMRGBlock(env);

			switch (sys->side) {
				case 'L':
					env = copyDMRGBlock(saved_blocksR[L - sys->length - 3]);
					break;

				case 'R':
					env = copyDMRGBlock(saved_blocksL[L - sys->length - 3]);
					break;
			}

			// Switch sides if at the end of the chain
			if (env->length == 1) {
				DMRGBlock *tempBlock = sys;
				sys = env;
				env = tempBlock;
			}

			printGraphic(sys, env);
			DMRGBlock *newSys = single_step(sys, env, m, 0);
			freeDMRGBlock(sys);
			sys = newSys;

			// Save new block
			switch (sys->side) {
				case 'L':
					if (saved_blocksL[sys->length-1]) { freeDMRGBlock(saved_blocksL[sys->length-1]); }
					saved_blocksL[sys->length-1] = copyDMRGBlock(sys);
					break;

				case 'R':
					if (saved_blocksR[sys->length-1]) { freeDMRGBlock(saved_blocksR[sys->length-1]); }
					saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
					break;
			}

			// Check if sweep is done
			if (sys->side == 'L' && 2 * sys->length == L) {
				break;
			}
		}
	}

	for (i = 0; i < L-4; i++) {
		if (saved_blocksL[i]) { freeDMRGBlock(saved_blocksL[i]); }
		if (saved_blocksR[i]) { freeDMRGBlock(saved_blocksR[i]); }
	}
	mkl_free(saved_blocksL);
	mkl_free(saved_blocksR);

	freeDMRGBlock(env);
	freeDMRGBlock(sys);
}
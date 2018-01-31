#ifndef DMRG_H
#define DMRG_H

#include "block.h"
#include "meas.h"
#include "params.h"

typedef struct {
	int abelianSectorize; // option determines whether to use abelian mz symmetry to only search for the ground state where mz = target_mz
						  // NOTE: if this is turned off it cannot really be turned on for later sweeps
	int target_mz; // mz for ground state if symmetry is active

	MAT_TYPE ** psi0_guessp; // pointer to guess for ground state. psi0_guessp is NULL if there is no guess and no guess tracking. 
	                         // *psi0_guessp is NULL when there is no guess but guess is returned.

	int measure; // option determines if you take measurements and return the results in meas
	meas_data_t *meas; // struct to hold measurements that may be returned by single_step


} dmrg_step_params_t;

DMRGBlock *single_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, dmrg_step_params_t *step_params);

meas_data_t *inf_dmrg(sim_params_t *params);

meas_data_t *fin_dmrg(sim_params_t *params);

#endif
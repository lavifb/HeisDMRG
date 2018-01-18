#ifndef DMRG_H
#define DMRG_H

#include "block.h"
#include "meas.h"
#include "params.h"

typedef struct {
	int target_mz; // mz for ground state if symmetry is active

	MAT_TYPE ** psi0_guessp; // pointer to guess for ground state. psi0_guessp is NULL if there is no guess and no guess tracking. 
	                         // *psi0_guessp is NULL when there is no guess but guess is returned.

	double tau; // time advance for TDMRG (set to 0 for normal dmrg)

} dmrg_step_params_t;

DMRGBlock *single_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, dmrg_step_params_t *step_params);

meas_data_t *meas_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, dmrg_step_params_t *step_params);

meas_data_t *inf_dmrg(sim_params_t *params);

meas_data_t *fin_dmrg(sim_params_t *params);

// Depreciated fin_dmrg with reflection symmetry
// meas_data_t *fin_dmrgR(sim_params_t *params);

#endif
#ifndef DMRG_H
#define DMRG_H

#include "block.h"
#include "meas.h"
#include "params.h"

DMRGBlock *single_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz, MAT_TYPE **const psi0_guessp);

meas_data_t *meas_step(const DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz, MAT_TYPE **const psi0_guessp);

void inf_dmrg(sim_params_t *params);

meas_data_t *fin_dmrg(sim_params_t *params);

meas_data_t *fin_dmrgR(sim_params_t *params);

#endif
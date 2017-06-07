#ifndef DMRG_H
#define DMRG_H

#include "block.h"
#include "meas.h"

void printGraphic(DMRGBlock *sys, DMRGBlock *env);

DMRGBlock *single_step(DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz);

meas_data_t *meas_step(DMRGBlock *sys, const DMRGBlock *env, const int m, const int target_mz);

void inf_dmrg(const int L, const int m, ModelParams *model);

void fin_dmrg(const int L, const int m_inf, const int num_sweeps, int *ms, ModelParams *model);

void fin_dmrgR(const int L, const int m_inf, const int num_sweeps, int *ms, ModelParams *model);

#endif
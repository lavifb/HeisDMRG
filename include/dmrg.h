#ifndef DMRG_H
#define DMRG_H

#include "block.h"

void single_step(DMRGBlock *sys, const DMRGBlock *env, const int m);

DMRGBlock *inf_dmrg(const int L, const int m, ModelParams *model);

#endif
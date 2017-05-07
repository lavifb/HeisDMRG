#ifndef DMRG_H
#define DMRG_H

#include "block.h"

DMRGBlock *single_step(DMRGBlock *sys, DMRGBlock *env, const int m);

DMRGBlock *inf_dmrg(const int L, const int m, ModelParams *model);

#endif
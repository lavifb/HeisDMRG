#ifndef INPUT_PARSE_H
#define INPUT_PARSE_H

#include "model.h"

typedef struct {
	int L;       // length of DMRG chain
	int minf;    // truncation dimension size when building system
	int num_ms;  // number of dmrg sweeps
	int *ms;     // truncation dimension size for each sweep

	ModelParams *model; // model params for the simulation

} sim_params_t;

int parseInputFile(const char *filename, sim_params_t *params);

#endif
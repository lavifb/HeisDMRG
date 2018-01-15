#ifndef PARAMS_H
#define PARAMS_H

#include "linalg.h"
#include "model.h"
#include <stdio.h>
#include <time.h>

typedef struct {
	int L;       // length of DMRG chain
	int minf;    // truncation dimension size when building system
	int num_ms;  // number of dmrg sweeps
	int *ms;     // truncation dimension size for each sweep

	model_t *model; // model params for the simulation
	int save_blocks; // 1 if you save blocks to disk during runtime and 0 to do the calculation entirely in RAM

	time_t *start_time;
	time_t *end_time;
	double runtime;

	char block_dir[1024]; // file path for saving blocks

} sim_params_t;

void printSimParams(FILE *stream, const sim_params_t *params);

void freeParams(sim_params_t *params);

#endif
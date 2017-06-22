#ifndef INPUT_PARSE_H
#define INPUT_PARSE_H

#include "model.h"

typedef struct {
	int L;       // length of DMRG chain
	int minf;    // truncation dimension size when building system
	int num_ms;  // number of dmrg sweeps
	int *ms;     // truncation dimension size for each sweep

	model_t *model; // model params for the simulation

} sim_params_t;

// ANSI colors for print output
#define TERM_RED     "\x1b[31m"
#define TERM_RESET   "\x1b[0m"

#define errprintf(M, ...) printf( TERM_RED "[ERROR] " M TERM_RESET , ##__VA_ARGS__)

int parseInputFile(const char *filename, sim_params_t *params);

void printSimParams(const sim_params_t *params);

#endif
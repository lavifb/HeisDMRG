#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include <mkl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	if (argc < 2) {
		errprintf("No input file specified!\n");
		return -1;
	}

	printf("Loading input file '%s'.\n\n", argv[1]);

	sim_params_t params = {};
	params.model = newNullModel();

	int status;
	status = parseInputFile(argv[1], &params);
	if (status < 0) {
		printf("Error parsing input file...\n");
		return status;
	}

	printf( "\n"
			"Heisenberg DMRG\n"
			"******************************\n\n");

	printSimParams(&params);

	printf( "\n"
			"******************************\n\n");

	model_t *model = params.model;
	compileParams(model);

	// inf_dmrg(L, m, model);

	// int ms[NUM_MS] = {10, 10, 10, 30, 30, 40, 40, 40};
	// int ms[1] = {5};

	// fin_dmrgR(20, 10, NUM_MS, ms, model);
	fin_dmrgR(params.L, params.minf, params.num_ms, params.ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

	freeModel(model);

	return 0;
}
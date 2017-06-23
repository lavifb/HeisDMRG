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

	printSimParams(&params);

	model_t *model = params.model;
	compileParams(model);

	// inf_dmrg(L, m, model);
	fin_dmrgR(params.L, params.minf, params.num_ms, params.ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

	freeModel(model);

	return 0;
}
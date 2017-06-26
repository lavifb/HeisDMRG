#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include <mkl.h>
#include <time.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	if (argc < 2) {
		errprintf("No input file specified!\n");
		return -1;
	}

	printf("Loading input file '%s'.\n\n", argv[1]);

	sim_params_t params = {};
	params.model = newNullModel();
	params.runtime = 0;

	int status;
	status = parseInputFile(argv[1], &params);
	if (status < 0) {
		printf("Error parsing input file...\n");
		return status;
	}

	model_t *model = params.model;
	compileParams(model);

	// Record start time
	time_t start_time = time(NULL);
	params.start_time = &start_time;

	printSimParams(stdout, &params);

	meas_data_t *meas;

	// Start cpu timer
	clock_t t_start = clock();

	// inf_dmrg(params.L, params.minf, model);
	meas = fin_dmrgR(params.L, params.minf, params.num_ms, params.ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

	// Record end time
	clock_t t_end = clock();
	params.runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printSimParams(stdout, &params);

	// outputMeasData("test-", meas);
	freeMeas(meas);

	freeModel(model);
	freeParams(&params);

	MKL_Free_Buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);
	if (nbytes_alloc > 0) {
		errprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
	}

	return 0;
}
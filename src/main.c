#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
#include <mkl.h>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>

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

	// file path for output dir
	char out_dir[1024];
	sprintf(out_dir, "L%d_M%d_sim_%d/", params.L, params.ms[params.num_ms-1], *params.start_time);

	mkdir(out_dir, 0755);

	// open file to log energies and truncation errors
	char log_filename[1024];
	sprintf(log_filename, "%senergies.log", out_dir);
	f_log = fopen(log_filename, "w");
	if (f_log == NULL) {
		errprintf("Cannot open file '%s'.\n", log_filename);
		return -1;
	}
	fprintf(f_log, "%-5s%-20s%-20s\n"
				"---------------------------------------------\n"
				 , "L", "Energy", "Truncation Error");


	printSimParams(stdout, &params);

	meas_data_t *meas;

	// Start cpu timer
	clock_t t_start = clock();

	// inf_dmrg(params.L, params.minf, model);
	// meas = fin_dmrg(params.L, params.minf, params.num_ms, params.ms, model);
	meas = fin_dmrgR(params.L, params.minf, params.num_ms, params.ms, model);

	// Record end time
	clock_t t_end = clock();
	params.runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("\n\nSimulation finished in %.3f seconds.\n", params.runtime);

	fclose(f_log);

	// Save sim params to a log file
	char plog_filename[1024];
	sprintf(plog_filename, "%sparams.log", out_dir); 

	FILE *plog_f = fopen(plog_filename, "w");
	if (plog_f == NULL) {
		errprintf("Cannot open file '%s'.\n", plog_filename);
		return -1;
	}

	printSimParams(plog_f, &params);
	fclose(plog_f);

	// Save Measurements
	outputMeasData(out_dir, meas);
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
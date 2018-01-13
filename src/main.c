#include "model.h"
#include "params.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
#include "util.h"
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

	sim_params_t *params = mkl_calloc(sizeof(sim_params_t), 1, MEM_DATA_ALIGN);
	params->model = newNullModel();
	params->runtime = 0;
	params->model->H_params = mkl_malloc(2 * sizeof(double), MEM_DATA_ALIGN);

	int status;
	status = parseInputFile(argv[1], params);
	if (status < 0) {
		printf("Error parsing input file...\n");
		return status;
	}

	model_t *model = params->model;
	compileParams(model);

	// Record start time
	time_t start_time = time(NULL);
	params->start_time = &start_time;
	params->end_time = NULL;

	// file path for output dir
	char out_dir[1024];
	sprintf(out_dir, "L%d_M%d_sim_%ld/", params->L, params->ms[params->num_ms-1], *params->start_time);

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


	printSimParams(stdout, params);

	meas_data_t *meas;

	printf("Running quick test on version "VERSION".\n\n");

	// Start cpu timer
	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	// inf_dmrg(params);
	// meas = fin_dmrg(params);
	meas = fin_dmrgR(params);

	// Record end time
	clock_gettime(CLOCK_MONOTONIC, &t_end);
	params->runtime = (t_end.tv_sec - t_start.tv_sec);
	params->runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("\n\nSimulation finished in %.3f seconds.\n", params->runtime);

	fclose(f_log);

	time_t end_time = time(NULL);
	params->end_time = &end_time;

	// Save sim params to a log file
	char plog_filename[1024];
	sprintf(plog_filename, "%sparams.log", out_dir); 

	FILE *plog_f = fopen(plog_filename, "w");
	if (plog_f == NULL) {
		errprintf("Cannot open file '%s'.\n", plog_filename);
		return -1;
	}

	printSimParams(plog_f, params);
	fclose(plog_f);

	// Save Measurements
	outputMeasData(out_dir, meas);
	freeMeas(meas);

	freeModel(model);
	freeParams(params);
	mkl_free(params);

	MKL_Free_Buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);
	if (nbytes_alloc > 0) {
		warnprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
	}

	return 0;
}
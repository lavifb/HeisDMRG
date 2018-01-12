#include "model.h"
#include "params.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "logio.h"
#include "util.h"
#include <mkl.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {

	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	int mm   = 20;
	int n_ms = 8;

	if (argc > 1) {
		mm = atoi(argv[1]);
		if (mm <= 0) {
			errprintf("Basis size '%d' must be positive.", mm);
		}
	}

	if (argc > 2) {
		n_ms = atoi(argv[2]);
		if (n_ms <= 0) {
			errprintf("Number of sweeps '%d' must be positive.", n_ms);
		}
	}

	sim_params_t *params = mkl_calloc(sizeof(sim_params_t), 1, MEM_DATA_ALIGN);
	params->L      = 32;
	params->minf   = mm;
	params->num_ms = n_ms;
	params->ms     = mkl_malloc(n_ms * sizeof(int), MEM_DATA_ALIGN);
	for (int i = 0; i < n_ms; i++) { params->ms[i] = mm; }

	model_t *model = newHeis2Model();
	model->fullLength = params->L;
	compileParams(model);

	params->model = model;

	printf("Running time test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	meas_data_t *meas = fin_dmrgR(params);
	// meas_data_t *meas = fin_dmrg(params);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("M=%d finished in %.3f seconds.\n\n", mm, runtime);

	int success = 0;

	freeMeas(meas);
	freeModel(model);
	freeParams(params);
	mkl_free(params);

	MKL_Free_Buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc, nbytes_alloc_peak;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);
	if (nbytes_alloc > 0) {
		errprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
		success = -1;
	}

	nbytes_alloc_peak = mkl_peak_mem_usage(MKL_PEAK_MEM);
	printf("Peak memory used is %lld bytes.\n", nbytes_alloc_peak);

	return 0;
}
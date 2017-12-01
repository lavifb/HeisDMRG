#include "model.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
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

	#define L    32
	int minf;
	int *ms = mkl_malloc(n_ms * sizeof(int), MEM_DATA_ALIGN);

	model_t *model = newHeis2Model();
	model->fullLength = L;

	compileParams(model);

	minf = mm;
	for (int i = 0; i < n_ms; i++) { ms[i] = mm; }

	printf("Running time test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	// meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);
	meas_data_t *meas = fin_dmrg(L, minf, n_ms, ms, model);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("M=%d finished in %.3f seconds.\n\n", mm, runtime);

	int success = 0;

	printf("Energy: %f\n", meas->energy);

	freeMeas(meas);
	freeModel(model);
	mkl_free(ms);

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
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
	#define N    2
	int minf;
	int ms[n_ms];

	model_t *model = newNullModel();
	model->d_model = N;
	model->J  = 1;
	model->Jz = 1;

	#if COMPLEX
	#include <complex.h>

	complex double H1[N*N] = { 0 , 0,
					    	   0 , 0 };
	complex double Sz[N*N] = { .5, 0,
					     	   0 ,-.5};
	complex double Sp[N*N] = { 0 , 1,
							   0 , 0 };
	#else
	MAT_TYPE H1[N*N] = { 0 , 0,
					     0 , 0 };
	MAT_TYPE Sz[N*N] = { .5, 0,
					     0 ,-.5};
	MAT_TYPE Sp[N*N] = { 0 , 1,
					     0 , 0 };
	#endif

	model->H1 = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->H1, H1, N*N * sizeof(MAT_TYPE));
	model->Sz = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sz, Sz, N*N * sizeof(MAT_TYPE));
	model->Sp = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sp, Sp, N*N * sizeof(MAT_TYPE));

	compileParams(model);

	int i;

	minf = mm;
	for (i = 0; i < n_ms; i++) { ms[i] = mm; }

	printf("Running time test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	// clock_t t_start = clock();
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);
	// meas_data_t *meas = fin_dmrg(L, minf, n_ms, ms, model);

	// clock_t t_end = clock();
	clock_gettime(CLOCK_MONOTONIC, &t_end);
	// double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("M=%d finished in %.3f seconds.\n\n", mm, runtime);

	int success = 0;

	freeMeas(meas);
	freeModel(model);

	MKL_Free_Buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);
	if (nbytes_alloc > 0) {
		errprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
		success = -1;
	}

	return 0;
}
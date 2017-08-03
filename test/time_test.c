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

	int mm;

	if (argc < 2) {
		mm = 20;
	} else {
		mm = atoi(argv[1]);
	}

	#define L    32
	#define n_ms 8
	#define N    2
	int minf;
	int ms[n_ms];

	model_t *model = newNullModel();
	model->d_model = N;
	model->J  = 1;
	model->Jz = 1;

	double H1[N*N] = { 0, 0,
					 0, 0 };
	double Sz[N*N] = {.5, 0,
					 0,-.5};
	double Sp[N*N] = { 0, 1,
					 0, 0 };

	model->H1 = mkl_malloc(N*N * sizeof(double), MEM_DATA_ALIGN);
	memcpy(model->H1, H1, N*N * sizeof(double));
	model->Sz = mkl_malloc(N*N * sizeof(double), MEM_DATA_ALIGN);
	memcpy(model->Sz, Sz, N*N * sizeof(double));
	model->Sp = mkl_malloc(N*N * sizeof(double), MEM_DATA_ALIGN);
	memcpy(model->Sp, Sp, N*N * sizeof(double));

	compileParams(model);

	int i;

	minf = mm;
	for (i = 0; i < n_ms; i++) { ms[i] = mm; }

	clock_t t_start = clock();

	meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);
	// meas_data_t *meas = fin_dmrg(L, minf, n_ms, ms, model);

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

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
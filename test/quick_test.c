#include "model.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
#include <mkl.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int main() {

	#define N    2
	#define L    100
	#define minf 10
	#define n_ms 3
	int ms[n_ms] = {10, 10, 20};

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

	// f_log = fopen("test_log.log", "w");
	// if (f_log == NULL) {
	//	errprintf("Cannot open file '%s'.\n", "test_log.log");
	//	return -1;
	// }
	// fprintf(f_log, "%-5s%-20s%-20s\n"
	//			"---------------------------------------------\n"
	//			 , "L", "Energy", "Truncation Error");

	clock_t t_start = clock();

	meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("Quick Test finished in %.3f seconds.\n\n", runtime);

	int success = 0;

	// Expected test results
	#define ETE  -.441271
	#define Sz13 -0x1.ebp-52
	#define Sz20 0x1.bap-49
	#define SS6  -0x1.c1f45da2c588p-9
	#define SS43 0x1.8fb3d2733c562p-6

	#define TOLERANCE 1e-5

	if (fabs(meas->energy - ETE) < TOLERANCE) {
		printf( TERM_GREEN "Energy Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Energy Test Failed!\nExpected Energy: %.17f\nMeasured Energy: %.17f\n",
			ETE, meas->energy);
		success = -1;
	}

	if (fabs(meas->Szs[13] - Sz13) < TOLERANCE && fabs(meas->Szs[20] - Sz20) < TOLERANCE) {
		printf( TERM_GREEN "Sz Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Sz Test Failed!\n");
		success = -1;
	}

	if (fabs(meas->SSs[6] - SS6) < TOLERANCE && fabs(meas->SSs[43] - SS43) < TOLERANCE) {
		printf( TERM_GREEN "SS correleation Test Passed!\n" TERM_RESET );
	} else {
		errprintf("SS correleation Test Failed!\n");
		success = -1;
	}

	// fclose(f_log);
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

	return success;
}
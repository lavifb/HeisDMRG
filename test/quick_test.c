#include "model.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
#include "matio.h"
#include <mkl.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

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

	printf("Running quick test on version "VERSION".\n\n");

	clock_t t_start = clock();

	meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("Quick Test finished in %.3f seconds.\n\n", runtime);


	// Getting path of binary
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) == NULL) {
		errprintf("getcwd() error. Cannot check solution.\n");
	}
	char path[1024];
	sprintf(path, "%s/%s", cwd, argv[0]);
	// remove filename and go up 1 dir
	char *pathpos = strrchr(path, '/');
	if (pathpos != NULL) {
	   *pathpos = '\0';
	} pathpos = strrchr(path, '/');
	if (pathpos != NULL) {
	   *pathpos = '\0';
	}


	int success = 0;

	// Expected test results
	#define ETE  -.441271
	// #define Sz13 -0x1.ebp-52
	// #define Sz20 0x1.bap-49
	// #define SS6  -0x1.c1f45da2c588p-9
	// #define SS43 0x1.8fb3d2733c562p-6

	#define TOLERANCE 1e-5

	if (fabs(meas->energy - ETE) < TOLERANCE) {
		printf( TERM_GREEN "Energy Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Energy Test Failed!\nExpected Energy: %.17f\nMeasured Energy: %.17f\n",
			ETE, meas->energy);
		success = -1;
	}

	const int n_sites = meas->num_sites;

	char path_Szs[1024];
	char path_SSs[1024];
	sprintf(path_Szs, "%s/test/quick_test_Szs.dat", path);
	sprintf(path_SSs, "%s/test/quick_test_SSs.dat", path);

	MAT_TYPE *test_Szs = mkl_malloc(n_sites * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *test_SSs = mkl_malloc(n_sites * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	readMat(path_Szs, test_Szs, n_sites);
	readMat(path_SSs, test_SSs, n_sites);


	int mat_errs = 0;

	for (int i=0; i<n_sites; i++) {
		if (fabs(meas->Szs[i] - test_Szs[i]) > TOLERANCE) {
			mat_errs++;
		}
	}
	if (mat_errs == 0) {
		printf( TERM_GREEN "Sz Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Sz Test Failed! %d/%d values incorrect.\n", mat_errs, n_sites);
	}

	mat_errs = 0;

	for (int i=0; i<n_sites; i++) {
		if (fabs(meas->SSs[i] - test_SSs[i]) > TOLERANCE) {
			mat_errs += 1;
		}
	}
	if (mat_errs == 0) {
		printf( TERM_GREEN "SS Test Passed!\n" TERM_RESET );
	} else {
		errprintf("SS Test Failed! %d/%d values incorrect.\n", mat_errs, n_sites);
	}

	// if (fabs(meas->Szs[13] - Sz13) < TOLERANCE && fabs(meas->Szs[20] - Sz20) < TOLERANCE) {
	// 	printf( TERM_GREEN "Sz Test Passed!\n" TERM_RESET );
	// } else {
	// 	errprintf("Sz Test Failed!\n");
	// 	success = -1;
	// }

	// if (fabs(meas->SSs[6] - SS6) < TOLERANCE && fabs(meas->SSs[43] - SS43) < TOLERANCE) {
	// 	printf( TERM_GREEN "SS correleation Test Passed!\n" TERM_RESET );
	// } else {
	// 	errprintf("SS correleation Test Failed!\n");
	// 	success = -1;
	// }

	// saveMat("quick_test_Szs.dat", meas->Szs, meas->num_sites);
	// saveMat("quick_test_SSs.dat", meas->SSs, meas->num_sites);

	mkl_free(test_Szs);
	mkl_free(test_SSs);
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
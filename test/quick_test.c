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
#include <complex.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

	#define L    100
	#define minf 10
	#define n_ms 3
	int ms[n_ms] = {10, 10, 20};

	// model_t *model = newHeis2Model();
	model_t *model = newLadderHeis2Model(4);
	model->fullLength = L;

	compileParams(model);

	printf("Running quick test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("Quick Test finished in %.3f seconds.\n\n", runtime);


	// Getting path of binary
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) == NULL) {
		errprintf("getcwd() error. Cannot check solution.\n");
	}
	char path[1024];
	sprintf(path, "%s/%s", cwd, argv[0]);
	// remove filename
	char *pathpos = strrchr(path, '/');
	if (pathpos != NULL) {
	   *pathpos = '\0';
	}

	int success = 0;

	// Expected test result
	#define ETE  -.441271

	#define TOLERANCE 1e-4
	#define SZ_TOLERANCE 1e-3

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
	#if COMPLEX
	sprintf(path_Szs, "%s/../test/test_mats/zquick_test_Szs.dat", path);
	sprintf(path_SSs, "%s/../test/test_mats/zquick_test_SSs.dat", path);
	#else
	sprintf(path_Szs, "%s/../test/test_mats/quick_test_Szs.dat", path);
	sprintf(path_SSs, "%s/../test/test_mats/quick_test_SSs.dat", path);
	#endif

	double *test_Szs = mkl_malloc(n_sites * sizeof(double), MEM_DATA_ALIGN);
	double *test_SSs = mkl_malloc(n_sites * sizeof(double), MEM_DATA_ALIGN);
	dreadMat(path_Szs, test_Szs, n_sites);
	dreadMat(path_SSs, test_SSs, n_sites);


	int mat_errs = 0;

	for (int i=0; i<n_sites; i++) {
		if (fabs(meas->Szs[i] - test_Szs[i]) > SZ_TOLERANCE) {
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
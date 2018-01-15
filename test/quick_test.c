#include "model.h"
#include "params.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "logio.h"
#include "matio.h"
#include "util.h"
#include <mkl.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

	sim_params_t *params = mkl_calloc(sizeof(sim_params_t), 1, MEM_DATA_ALIGN);
	params->L      = 100;
	params->minf   = 10;
	#define n_ms 3

	params->num_ms = n_ms;
	int ms[n_ms] = {10, 10, 20};
	params->ms     = ms;

	model_t *model = newHeis2Model();
	compileParams(model);
	params->model  = model;
	params->save_blocks = 0; // do not save blocks to disk

	model->fullLength = params->L;

	time_t start_time = time(NULL);
	// file path for output dir
	sprintf(params->block_dir, "temp-L%d_M%d_sim_%ld", params->L, params->ms[params->num_ms-1], start_time);
	mkdir(params->block_dir, 0755);

	printf("Running quick test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	meas_data_t *meas = fin_dmrg(params);
	// meas_data_t *meas = fin_dmrgR(params);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("Quick Test finished in %.3f seconds.\n\n", runtime);


	// Delete temporary files
	rmrf(params->block_dir);

	// Getting path of binary
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) == NULL) {
		errprintf("getcwd() error. Cannot check solution.\n");
		exit(1);
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
		failprintf("Energy Test Failed!\nExpected Energy: %.17f\nMeasured Energy: %.17f\n",
			ETE, meas->energy);
		success = 1;
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
		passprintf("Sz Test Passed!\n");
	} else {
		failprintf("Sz Test Failed! %d/%d values incorrect.\n", mat_errs, n_sites);
		success = 1;
	}

	mat_errs = 0;

	for (int i=0; i<n_sites; i++) {
		if (fabs(meas->SSs[i] - test_SSs[i]) > TOLERANCE) {
			mat_errs += 1;
		}
	}
	if (mat_errs == 0) {
		passprintf("SS Test Passed!\n");
	} else {
		failprintf("SS Test Failed! %d/%d values incorrect.\n", mat_errs, n_sites);
		success = 1;
	}

	// saveMat("quick_test_Szs.dat", meas->Szs, meas->num_sites);
	// saveMat("quick_test_SSs.dat", meas->SSs, meas->num_sites);

	mkl_free(test_Szs);
	mkl_free(test_SSs);
	freeMeas(meas);
	freeModel(model);
	mkl_free(params);

	MKL_Free_Buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);
	if (nbytes_alloc > 0) {
		warnprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
		success = -1;
	}

	return success;
}
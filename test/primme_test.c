#include "model.h"
#include "params.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "dmrg.h"
#include "logio.h"
#include "matio.h"
#include "linalg.h"
#include "util.h"
#include <mkl.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

	int runs = 100;

	if (argc > 1) {
		runs = atoi(argv[1]);
	}

	#define N 116

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

	// paths for test matrices
	char path_Hs[1024];
	char path_psi0[1024];
	#if COMPLEX
	sprintf(path_Hs,   "%s/../test/test_mats/zprimme_test_Hs.dat", path);
	sprintf(path_psi0, "%s/../test/test_mats/zprimme_test_psi0.dat", path);
	#else
	sprintf(path_Hs,   "%s/../test/test_mats/primme_test_Hs.dat", path);
	sprintf(path_psi0, "%s/../test/test_mats/primme_test_psi0.dat", path);
	#endif

	MAT_TYPE *Hs_r = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *psi0 = mkl_malloc(N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	int info;

	info = readMat(path_Hs, Hs_r, N*N);
	if (info < 0) {
		errprintf("Could not load matrix 'Hs' from '%s'\n", path_Hs);
		return -1;
	}
	readMat(path_psi0, psi0, N);
	if (info < 0) {
		errprintf("Could not load matrix 'psi0' from '%s'\n", path_psi0);
		return -1;
	}

	MAT_TYPE **psi0s = mkl_malloc(runs * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

	for (int i=0; i<runs; i++) {
		psi0s[i] = mkl_malloc(N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(psi0s[i], psi0, N*sizeof(MAT_TYPE));
	}

	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	printf("Running dprimme test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	for (int i=0; i<runs; i++) {
		primmeWrapper(Hs_r, N, energies, psi0s[i], 1, 0);
	}

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("dprimme test finished %d runs in %.3f seconds.\n\n", runs, runtime);

	int success = 0;

	#define TOLERANCE 1e-5

	#if COMPLEX
	#include <complex.h>
	complex double *psi0z   = (complex double *) psi0;
	complex double **psi0sz = (complex double **) psi0s;
	#endif

	int mat_errs = 0;
	for (int i=0; i<N; i++) {
		#if COMPLEX
		if (cabs(psi0z[i])-cabs(psi0sz[runs-1][i]) > TOLERANCE) {
		#else
		if (fabs(psi0[i])-fabs(psi0s[runs-1][i]) > TOLERANCE) {
		#endif
			mat_errs++;
		}
	}

	if (mat_errs == 0) {
		printf( TERM_GREEN "dprimme Test Passed!\n" TERM_RESET );
	} else {
		errprintf("dprimme Test Failed! %d/%d values incorrect.\n", mat_errs, N);
		success = -1;
	}

	// saveMat("psi0z.dat", psi0s[runs-1], N);

	mkl_free(Hs_r);
	mkl_free(psi0);
	for (int i=0; i<runs; i++) { mkl_free(psi0s[i]); }
	mkl_free(psi0s);
	mkl_free(energies);


	return success;
}
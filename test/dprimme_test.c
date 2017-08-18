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

int main(int argc, char *argv[]) {

	int runs = 100;

	if (argc > 1) {
		runs = atoi(argv[1]);
	}

	#define N 116

	MAT_TYPE *Hs_r = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	MAT_TYPE *psi0 = mkl_malloc(N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

	readMat("Hs.dat", Hs_r, N*N);
	readMat("psi0.dat", psi0, N);

	MAT_TYPE **psi0s = mkl_malloc(runs * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);

	for (int i=0; i<runs; i++) {
		psi0s[i] = mkl_malloc(N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		memcpy(psi0s[i], psi0, N*sizeof(MAT_TYPE));
	}

	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	printf("Running dprimme test on version "VERSION".\n\n");

	clock_t t_start = clock();

	for (int i=0; i<runs; i++) {
		primmeWrapper(Hs_r, N, energies, psi0s[i], 1);
	}

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("dprimme test finished %d runs in %.3f seconds.\n\n", runs, runtime);

	int success = 0;

	#define TOLERANCE 1e-5

	int mat_errs = 0;
	for (int i=0; i<N; i++) {
		if (fabs(psi0[i])-fabs(psi0s[runs-1][i]) > TOLERANCE) {
			mat_errs++;
		}
	}

	if (mat_errs == 0) {
		printf( TERM_GREEN "dprimme Test Passed!\n" TERM_RESET );
	} else {
		errprintf("dprimme Test Failed! %d/%d values incorrect.\n", mat_errs, N);
		success = -1;
	}

	mkl_free(Hs_r);
	mkl_free(psi0);
	for (int i=0; i<runs; i++) { mkl_free(psi0s[i]); }
	mkl_free(psi0s);
	mkl_free(energies);


	return success;
}
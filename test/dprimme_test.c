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

	double *Hs_r = mkl_malloc(N*N * sizeof(double), MEM_DATA_ALIGN);
	double *psi0 = mkl_malloc(N * sizeof(double), MEM_DATA_ALIGN);

	readMat("Hs.dat", Hs_r, N*N);
	readMat("psi0.dat", psi0, N);

	double **psi0s = mkl_malloc(runs * sizeof(double *), MEM_DATA_ALIGN);

	for (int i=0; i<runs; i++) {
		psi0s[i] = mkl_malloc(N * sizeof(double), MEM_DATA_ALIGN);
		memcpy(psi0s[i], psi0, N*sizeof(double));
	}


	double *energies = mkl_malloc(sizeof(double), MEM_DATA_ALIGN);

	clock_t t_start = clock();

	for (int i=0; i<runs; i++) {
		primmeWrapper(Hs_r, N, energies, psi0s[i], 1);
	}

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("dprimme test finished %d runs in %.3f seconds.\n\n", runs, runtime);

	int success = 0;

	#define TOLERANCE 1e-5

	for (int i=0; i<N; i++) {
		if (fabs(psi0[i])-fabs(psi0s[runs-1][i]) > TOLERANCE) {
			success = -1;
			errprintf("Result does not match! %.8f =/= %.8f\n", psi0[i], psi0s[runs-1][i]);
			return -1;
		}
	}

	mkl_free(Hs_r);
	mkl_free(psi0);
	// mkl_free(psi02);
	mkl_free(energies);


	return success;
}
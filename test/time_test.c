#include "model.h"
#include "block.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include "logio.h"
#include <mkl.h>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {

	int mm;

	if (argc < 2) {
		mm = 10;
	} else {
		mm = atoi(argv[1]);
	}

	#define L    100
	#define n_ms 8
	int minf;
	int ms[n_ms];

	model_t *model = newNullModel();
	model->d_model = 2;
	model->J  = 1;
	model->Jz = 1;

	double H1[4] = { 0, 0,
					 0, 0 };
	double Sz[4] = {.5, 0,
					 0,-.5};
	double Sp[4] = { 0, 1,
					 0, 0 };

	model->H1 = H1;
	model->Sz = Sz;
	model->Sp = Sp;

	compileParams(model);

	int i;

	minf = mm;
	for (i = 0; i < n_ms; i++) { ms[i] = mm; }

	clock_t t_start = clock();

	// meas_data_t *meas = fin_dmrgR(L, minf, n_ms, ms, model);
	meas_data_t *meas = fin_dmrg(L, minf, n_ms, ms, model);

	clock_t t_end = clock();
	double runtime = (double)(t_end - t_start) / CLOCKS_PER_SEC;

	printf("M=%d finished in %.3f seconds.\n\n", mm, runtime);

	int success = 0;

	return 0;
}
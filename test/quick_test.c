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

int main() {

	#define L    100
	#define minf 10
	#define n_ms 3
	int ms[n_ms] = {10, 10, 20};

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
	#define ETE  -0x1.c3dc97de7fe89p-2
	#define Sz12 -0x1.ebp-52
	#define Sz19 0x1.bap-49
	#define SS5  -0x1.c1f45da2c588p-9
	#define SS42 0x1.8fb3d2733c562p-6

	if (meas->energy == ETE) {
		printf( TERM_GREEN "Energy Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Energy Test Failed!\nExpected Energy: %.17f\nMeasured Energy: %.17f\n",
			ETE, meas->energy);
		success = -1;
	}

	if (meas->Szs[12] == Sz12 && meas->Szs[19] == Sz19) {
		printf( TERM_GREEN "Sz Test Passed!\n" TERM_RESET );
	} else {
		errprintf("Sz Test Failed!\n");
		success = -1;
	}

	if (meas->SSs[5] == SS5 && meas->SSs[42] == SS42) {
		printf( TERM_GREEN "SS correleation Test Passed!\n" TERM_RESET );
	} else {
		errprintf("SS correleation Test Failed!\n");
		success = -1;
	}

	// fclose(f_log);


	return success;
}
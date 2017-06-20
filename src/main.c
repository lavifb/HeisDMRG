#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include "input_parser.h"
#include <mkl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	if (argc < 2) {
		printf("No input file specified!\n");
		return -1;
	}

	printf("Loading input file '%s'.\n", argv[1]);

	sim_params_t params = {};
	model_t sim_model;
	params.model = &sim_model;

	int status;
	status = parseInputFile(argv[1], &params);
	if (status < 0) {
		printf("Error parsing input file...\n");
		return status;
	}

	int L    = params.L;
	int minf = params.minf;
	int *ms  = params.ms;


	printf( "\n\n"
			"Heisenberg DMRG\n"
			"******************************\n\n");

	printSimParams(&params);

	printf( "\n"
			"******************************\n\n");

	model_t *model = params.model;

	#define N 2
	model->d_model = N;
	model->J  = 1;
	model->Jz = 1;
	model->num_ops = 3;

	// One site matrices
	double H1[N*N] = {0.0, 0.0, 
	                  0.0, 0.0};
	double Sz[N*N] = {0.5, 0.0, 
	                  0.0,-0.5};
	double Sp[N*N] = {0.0, 0.0, 
	                  1.0, 0.0};
	double Id[N*N] = {1.0, 0.0, 
	                  0.0, 1.0};

	int mzs[N] = {1, -1};

	model->H1 = H1;
	model->Sz = Sz;
	model->Sp = Sp;
	model->Id = Id;

	double *init_ops[3];
	model->init_ops = init_ops;
	model->init_ops[0] = H1;
	model->init_ops[1] = Sz;
	model->init_ops[2] = Sp;

	model->init_mzs = mzs;

	// inf_dmrg(L, m, model);

	// #define NUM_MS 1

	// int ms[NUM_MS] = {10};
	// int ms[NUM_MS] = {10, 10, 10, 30, 30, 40, 40, 40};
	// int ms[1] = {5};

	// fin_dmrgR(20, 10, NUM_MS, ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

}
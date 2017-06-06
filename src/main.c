#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include <mkl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	printf("Heisenberg DMRG\n");

	int L = 1000;
	int m = 30;

	ModelParams *model = (ModelParams *)mkl_malloc(sizeof(ModelParams), MEM_DATA_ALIGN);

	#define N 2
	model->d_model = N;
	model->J  = 1;
	model->Jz = 1;
	model->num_ops = 3;

	// One site matrices
	double H1[N*N] = {0.0, 0.0, 
	                  0.0,  0.0};
	double Sz[N*N] = {0.5, 0.0, 
	                  0.0, -0.5};
	double Sp[N*N] = {0.0, 0.0, 
	                  1.0,  0.0};
	double Id[N*N] = {1.0, 0.0, 
	                  0.0,  1.0};

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

	#define NUM_MS 8

	// int ms[NUM_MS] = {20};
	int ms[NUM_MS] = {10, 10, 10, 30, 30, 40, 40, 40};
	// int ms[1] = {5};

	fin_dmrgR(20, 10, NUM_MS, ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

	mkl_free(model);
}
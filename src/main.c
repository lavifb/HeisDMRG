#include "model.h"
#include "block.h"
#include "linalg.h"
#include "dmrg.h"
#include <mkl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	printf("Heisenberg DMRG\n");

	int L = 100;
	int m = 5;

	ModelParams *model = (ModelParams *)mkl_malloc(sizeof(ModelParams), MEM_DATA_ALIGN);

	#define N 2
	model->d_model = N;
	model->J  = 1;
	model->Jz = 1;
	model->num_ops = 3;

	// One site matrices
	double H1[N*N] = {0.0, 0.0, 0.0,  0.0};
	double Sz[N*N] = {0.5, 0.0, 0.0, -0.5};
	double Sp[N*N] = {0.0, 0.0, 1.0,  0.0};
	double Id[N*N] = {1.0, 0.0, 0.0,  1.0};

	model->H1 = H1;
	model->Sz = Sz;
	model->Sp = Sp;
	model->Id = Id;

	double *init_ops[3];
	model->init_ops = init_ops;
	model->init_ops[0] = H1;
	model->init_ops[1] = Sz;
	model->init_ops[2] = Sp;

	model->num_qns = 0;

	// inf_dmrg(L, m, model);

	#define NUM_MS 4

	// int ms[NUM_MS] = {10};
	int ms[NUM_MS] = {10, 20, 30, 40};
	// int ms[1] = {5};

	// fin_dmrg(20, 10, NUM_MS, ms, model);
	// fin_dmrg(10, 5, 1, ms, model);

	// print_matrix("Sp", 2, 2, Sp, 2);
	// double Sm[N*N];
	// mkl_domatcopy('C', 'c', N, N, 1.0, Sp, N, Sm, N); // Transpose Sp1 to Sm1
	// print_matrix("Sm1", N, N, Sm, N);

	// double trans[2] = {.707106781, .707106781};

	// print_matrix("trans", 1, 2, trans, 2);

	// print_matrix("Transformed", 1, 1, transformOp(2, 1, trans, Sp), 1);
}
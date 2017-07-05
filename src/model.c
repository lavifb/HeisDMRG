#include "model.h"
#include "linalg.h"
#include <mkl.h>

/*  Take model parameters given by input and compile them to be usable

	This includes transposing all matrices to ColMajor format and filling out
	unfilled parameters.
*/
void compileParams(model_t *model) {

	int dim = model->d_model;

	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->H1, dim, dim);
	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->Sz, dim, dim);
	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->Sp, dim, dim);
	model->Id = identity(dim);

	model->num_ops  = 3;
	model->init_ops = (double **)mkl_malloc(3 * sizeof(double *), MEM_DATA_ALIGN);
	model->init_ops[0] = model->H1;
	model->init_ops[1] = model->Sz;
	model->init_ops[2] = model->Sp;

	model->init_mzs = (int *)mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	int i;
	for (i=0; i<dim; i++) {
		// init_mzs stores 2*mz to make it an integer
		model->init_mzs[i] = model->Sz[i*dim+i] * 2;
	}

	// Set Hamiltonian parameters
	model->H_params = (double *)mkl_malloc(2 * sizeof(double), MEM_DATA_ALIGN);
	model->H_params[0] = model->J/2;
	model->H_params[1] = model->Jz;

	// Set Hamiltonian interaction function
	model->H_int = &HeisenH_int;
}

/* Nulls out model parameters
*/
model_t *newNullModel() {

	model_t *model = (model_t *)mkl_calloc(sizeof(model_t), 1, MEM_DATA_ALIGN);

	model->d_model = 0;
	model->H1 = NULL;
	model->Sz = NULL;
	model->Sp = NULL;
	model->Id = NULL;
	model->J  = 0;
	model->Jz = 0;
	model->init_mzs = NULL;
	model->num_ops = 0;
	model->init_ops = NULL;
	model->H_params = NULL;
	model->H_int = NULL;

	return model;
}

void freeModel(model_t *model) {

	if(model->H1) { mkl_free(model->H1); }
	if(model->Sz) { mkl_free(model->Sz); }
	if(model->Sp) { mkl_free(model->Sp); }
	if(model->Id) { mkl_free(model->Id); }
	if(model->init_mzs) { mkl_free(model->init_mzs); }
	if(model->init_ops) { mkl_free(model->init_ops); }
	if(model->H_params) { mkl_free(model->H_params); }

	mkl_free(model);
}

/*  Interaction part of Heisenberg Hamiltonian
	H_int = J/2 (kron(Sp1, Sm2) + kron(Sm1, Sp2)) + Jz kron(Sz1, Sz2)
*/
double *HeisenH_int(const double* H_params, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2) {
	int N = dim1*dim2; // size of new basis

	double *H_int = (double *)mkl_calloc(N*N,        sizeof(double), MEM_DATA_ALIGN);
	double *Sm1   = (double *)mkl_malloc(dim1*dim1 * sizeof(double), MEM_DATA_ALIGN);
	double *Sm2   = (double *)mkl_malloc(dim2*dim2 * sizeof(double), MEM_DATA_ALIGN);
	__assume_aligned(H_int,   MEM_DATA_ALIGN);
	__assume_aligned(Sp1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sp2  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm1  ,   MEM_DATA_ALIGN);
	__assume_aligned(Sm2  ,   MEM_DATA_ALIGN);

	double J2 = H_params[0]; // J/2
	double Jz = H_params[1];

	kron(Jz, dim1, dim2, Sz1, Sz2, H_int); // H_int += Jz * kron(Sz1, Sz2)

	mkl_domatcopy('C', 'C', dim1, dim1, 1.0, Sp1, dim1, Sm1, dim1); // Transpose Sp1 to Sm1
	mkl_domatcopy('C', 'C', dim2, dim2, 1.0, Sp2, dim2, Sm2, dim2); // Transpose Sp2 to Sm2

	kron(J2, dim1, dim2, Sp1, Sm2, H_int); // H_int += J/2 * kron(Sp1, Sm2)
	kron(J2, dim1, dim2, Sm1, Sp2, H_int); // H_int += J/2 * kron(Sm1, Sp2)


	mkl_free(Sm1);
	mkl_free(Sm2);

	return H_int;
}
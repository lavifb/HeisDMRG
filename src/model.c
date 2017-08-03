#include "model.h"
#include "hamil.h"
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
	for (int i=0; i<dim; i++) {
		// init_mzs stores 2*mz to make it an integer
		model->init_mzs[i] = model->Sz[i*dim+i] * 2;
	}

	// Set Hamiltonian parameters
	model->H_params = (double *)mkl_malloc(2 * sizeof(double), MEM_DATA_ALIGN);
	model->H_params[0] = model->J/2;
	model->H_params[1] = model->Jz;

	// Set Hamiltonian interaction function
	model->H_int   = &HeisenH_int;
	model->H_int_r = &HeisenH_int_r;
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
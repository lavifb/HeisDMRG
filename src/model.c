#include "model.h"
#include "hamil.h"
#include "linalg.h"
#include "input_parser.h"
#include <mkl.h>
#include <string.h>

/*  Take model parameters given by input and compile them to be usable

	This includes transposing all matrices to ColMajor format and filling out
	unfilled parameters.
*/
void compileParams(model_t *model) {

	int dim = model->d_model;


	#if COMPLEX
	const MKL_Complex16 one = {.real=1.0, .imag=0.0};
	mkl_zimatcopy('C', 'T', dim, dim, one, model->H1, dim, dim);
	mkl_zimatcopy('C', 'T', dim, dim, one, model->Sz, dim, dim);
	mkl_zimatcopy('C', 'T', dim, dim, one, model->Sp, dim, dim);
	#else
	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->H1, dim, dim);
	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->Sz, dim, dim);
	mkl_dimatcopy('C', 'T', dim, dim, 1.0, model->Sp, dim, dim);
	#endif

	model->Id = identity(dim);

	model->num_ops  = 2*model->ladder_width + 1;
	model->init_ops = mkl_malloc(model->num_ops * sizeof(MAT_TYPE *), MEM_DATA_ALIGN);
	model->init_ops[0] = model->H1;
	model->init_ops[1] = model->Sz;
	model->init_ops[2] = model->Sp;

	for (int i=1; i<model->ladder_width; i++) {
		model->init_ops[i*2 + 1] = model->H1;
		model->init_ops[i*2 + 2] = model->H1;
	}

	model->init_mzs = mkl_malloc(dim * sizeof(int), MEM_DATA_ALIGN);
	for (int i=0; i<dim; i++) {
		// init_mzs stores 2*mz to make it an integer
		#if COMPLEX
		model->init_mzs[i] = model->Sz[i*dim+i].real * 2;
		#else
		model->init_mzs[i] = model->Sz[i*dim+i] * 2;
		#endif
	}

	// TODO: don't copy init_ops into block (doesn't really matter though...)
	model->single_block = createDMRGBlock(model);

	// Set interaction Hamiltonian functions
	if (strcmp(model->geometry, "1D") == 0 || strcmp(model->geometry, "1d") == 0 || strcmp(model->geometry, "chain") == 0) {
		model->H_int = &HeisenH_int;
		#if USE_PRIMME
		model->H_int_mats = &HeisenH_int_mats;
		#else
		model->H_int_r = &HeisenH_int_r;
		#endif
	} else if (strcmp(model->geometry, "Ladder") == 0 || strcmp(model->geometry, "ladder") == 0) {
		model->H_int = &LadderH_int;
		#if USE_PRIMME
		model->H_int_mats = &LadderH_int_mats;
		#else
		model->H_int_r = &LadderH_int_r;
		#endif
	} else if (strcmp(model->geometry, "") == 0) {
		errprintf("No valid model geometry provided. Please provide a valid model geometry.\n");
		exit(1);
	} else {
		errprintf("'%s' is not a valid model geometry. Please provide a valid model geometry.\n", model->geometry);
		exit(1);
	}
}

/* Nulls out model parameters
*/
model_t *newNullModel() {

	model_t *model = mkl_calloc(sizeof(model_t), 1, MEM_DATA_ALIGN);

	model->d_model = 0;
	model->H1 = NULL;
	model->Sz = NULL;
	model->Sp = NULL;
	model->Id = NULL;
	model->init_mzs = NULL;
	model->num_ops = 0;
	model->init_ops = NULL;
	model->H_params = NULL;

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
	if(model->single_block) { freeDMRGBlock(model->single_block); }

	mkl_free(model);
}

/*  Function to create a model for use in tests.
*/
model_t *newHeis2Model() {

	model_t *model = mkl_calloc(sizeof(model_t), 1, MEM_DATA_ALIGN);

	#define N 2
	model->d_model = N;

	#if COMPLEX
	#include <complex.h>

	complex double H1[N*N] = { 0 , 0,
					    	   0 , 0 };
	complex double Sz[N*N] = { .5, 0,
					     	   0 ,-.5};
	complex double Sp[N*N] = { 0 , 1,
							   0 , 0 };
	#else
	MAT_TYPE H1[N*N] = { 0 , 0,
					     0 , 0 };
	MAT_TYPE Sz[N*N] = { .5, 0,
					     0 ,-.5};
	MAT_TYPE Sp[N*N] = { 0 , 1,
					     0 , 0 };
	#endif

	model->H1 = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->H1, H1, N*N * sizeof(MAT_TYPE));
	model->Sz = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sz, Sz, N*N * sizeof(MAT_TYPE));
	model->Sp = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sp, Sp, N*N * sizeof(MAT_TYPE));

	model->Id = NULL;
	model->init_mzs = NULL;
	model->num_ops = 0;
	model->init_ops = NULL;
	model->ladder_width = 1;
	strncpy(model->geometry, "1D", 15);

	double *H_params = mkl_malloc(2 * sizeof(double), MEM_DATA_ALIGN);
	H_params[0] = 1.0;
	H_params[1] = 1.0;
	model->H_params = H_params;

	return model;
}

/*  Function to create a ladder model for use in tests.
*/
model_t *newLadderHeis2Model(int ladder_width) {

	model_t *model = mkl_calloc(sizeof(model_t), 1, MEM_DATA_ALIGN);

	#define N 2
	model->d_model = N;

	#if COMPLEX
	#include <complex.h>

	complex double H1[N*N] = { 0 , 0,
					    	   0 , 0 };
	complex double Sz[N*N] = { .5, 0,
					     	   0 ,-.5};
	complex double Sp[N*N] = { 0 , 1,
							   0 , 0 };
	#else
	double H1[N*N] = { 0 , 0,
					   0 , 0 };
	double Sz[N*N] = { .5, 0,
					   0 ,-.5};
	double Sp[N*N] = { 0 , 1,
					   0 , 0 };
	#endif

	model->H1 = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->H1, H1, N*N * sizeof(MAT_TYPE));
	model->Sz = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sz, Sz, N*N * sizeof(MAT_TYPE));
	model->Sp = mkl_malloc(N*N * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
	memcpy(model->Sp, Sp, N*N * sizeof(MAT_TYPE));

	model->Id = NULL;
	model->init_mzs = NULL;
	model->num_ops = 0;
	model->init_ops = NULL;
	model->ladder_width = ladder_width;
	strncpy(model->geometry, "Ladder", 15);

	double *H_params = mkl_malloc(2 * sizeof(double), MEM_DATA_ALIGN);
	H_params[0] = 1.0;
	H_params[1] = 1.0;
	model->H_params = H_params;

	return model;
}
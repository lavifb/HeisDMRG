#ifndef MODEL_H
#define MODEL_H

typedef struct {
	int d_model; // single site basis size
	double *H1;  // single site Hamiltonian
	double *Sz;  // single site Sz
	double *Sp;  // single site S+
	double *Id;  // single site Identity Matrix
	int *init_mzs;	// 2*mz quantum number for each state
	int num_ops;
	double **init_ops; // single site block tracked operators
	
	double J;
	double Jz;
	double *H_params;
	// Pointer to interaction Hamiltonian
	double *(*H_int)(const double* H_params, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2);
	double *(*H_int_r)(const double* H_params, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2, 
					const int num_ind, const int *restrict inds);
} model_t;


void compileParams(model_t *model);

model_t *newNullModel();

void freeModel(model_t *model);

#endif
#ifndef MODEL_H
#define MODEL_H

typedef struct {
	int d_model; // single site basis size
	double *H1;  // single site Hamiltonian
	double *Sz;  // single site Sz
	double *Sp;  // single site S+
	double *Id;  // single site Identity Matrix
	double J;
	double Jz;
	int *init_mzs;	// 2*mz quantum number for each state
	int num_ops;
	double **init_ops; // single site block tracked operators
} model_t;


void compileParams(model_t *model);

model_t *newNullModel();

void freeModel(model_t *model);

double *HeisenH_int(const double J, const double Jz, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2);

#endif
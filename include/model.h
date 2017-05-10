#ifndef MODEL_H
#define MODEL_H

typedef struct {
	int d_model; // single site basis size
	double *H1; // single site Hamiltonian
	double *Sz; // single site Sz
	double *Sp; // single site S+
	double *Id; // single site Identity Matrix
	double J;
	double Jz;
	int num_ops;
	double **init_ops; // single site block tracked operators

	int num_qns;   	// number of quantum numbers
	int **init_qns;	// list of quantum numbers for single site 
	               	// each initqns[i] has size d_model)
} ModelParams;

double *HeisenH_int(const double J, const double Jz, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2);

#endif
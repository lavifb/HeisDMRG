#ifndef HAMIL_H
#define HAMIL_H

// #include "block.h"

double *HeisenH_int(const double* H_params, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2);

double *HeisenH_int_r(const double* H_params, const int dim1, const int dim2, 
					const double *restrict Sz1, const double *restrict Sp1, 
					const double *restrict Sz2, const double *restrict Sp2, 
					const int num_ind, const int *restrict inds);

#endif
#ifndef MODEL_H
#define MODEL_H

typedef struct {
    int dModel; // single site basis size
    double *H1; // single site Hamiltonian
    double *Sz; // single site Sz
    double *Sp; // single site S+
    double *Id; // single site Identity Matrix
    double J;
    double Jz;
} ModelParams;

double *HeisenH_int(const double J, const double Jz, const int dim1, const int dim2, 
                    const double *restrict Sz1, const double *restrict Sp1, 
                    const double *restrict Sz2, const double *restrict Sp2);

#endif
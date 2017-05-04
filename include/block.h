#ifndef BLOCK_H
#define BLOCK_H

// #include "uthash.h"

typedef struct {
	int dModel; // single site basis size
	double *H1; // single site Hamiltonian
	double *Sz; // single site Sz
	double *Sp; // single site S+
	double J;
	double Jz;
} ModelParams;

typedef struct {
	int length;
	int basis_size;
	double **ops;
	int num_ops;
	ModelParams *model;
} DMRGBlock;

#endif
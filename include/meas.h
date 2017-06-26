#ifndef MEAS_H
#define MEAS_H

#include <stdio.h>
#include <time.h>

typedef struct {
	int L;
	double energy;

	int num_sites;			// number of site spins measured
	double* Szs;  			// single site spins <S_i>
	double* SSs;  			// spin-spin corr <S_i S_j> where j is a middle spin
	// double* truncation_error;

} meas_data_t;

meas_data_t *createMeas(int num_sites);

void freeMeas(meas_data_t *meas);

int outputMeasData(const char* path, meas_data_t *meas);

#endif
#include "meas.h"
#include <mkl.h>

meas_data_t *createMeas(int num_sites) {

	meas_data_t *meas = (meas_data_t *)mkl_malloc(sizeof(meas_data_t), MEM_DATA_ALIGN);

	meas->num_sites = num_sites;
	meas->Szs = (double *)mkl_malloc(num_sites * sizeof(double), MEM_DATA_ALIGN);
	meas->SSs = (double *)mkl_malloc(num_sites * sizeof(double), MEM_DATA_ALIGN);

	return meas;
}

void freeMeas(meas_data_t *meas) {

	mkl_free(meas->Szs);
	mkl_free(meas->SSs);

	// if(meas->truncation_error) { mkl_free(meas->truncation_error); }

	mkl_free(meas);
}
#include "meas.h"

void freeMeas(meas_data_t *meas) {

	if(meas->Szs) { mkl_free(meas->Szs); }
	if(meas->SSs) { mkl_free(meas->SSs); }

	if(meas->truncation_error) { mkl_free(meas->truncation_error); }

	mkl_free(meas);
}
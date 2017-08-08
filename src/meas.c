#include "meas.h"
#include "input_parser.h"
#include "linalg.h"
#include <mkl.h>
#include <stdio.h>

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

/*  Write measure data to files in path
*/
int outputMeasData(const char* path, meas_data_t *meas) {
	
	char filename[1024];
	sprintf(filename, "%smeasurements.dat", path); 

	FILE *m_f = fopen(filename, "w");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	fprintf(m_f, "%-6s%-20s%-20s\n"
				 "---------------------------------------------\n"
				 , "Site", "Sz", "SS");

	for (int i = meas->num_sites-1; i>=0; i--) {
		// #if COMPLEX
		// fprintf(m_f, "%-6d%- 20.12f%- 20.12f\n", i+1, meas->Szs[i].real, meas->SSs[i].real);
		// #else
		fprintf(m_f, "%-6d%- 20.12f%- 20.12f\n", i+1, meas->Szs[i], meas->SSs[i]);
		// #endif
	}

	fclose(m_f);

	return 0;
}
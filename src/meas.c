#include "meas.h"
#include "input_parser.h"
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
	
	// log file
	char log_filename[1024];
	sprintf(log_filename, "%soutput.log", path); 

	FILE *log_f = fopen(log_filename, "w");
	if (log_f == NULL) {
		errprintf("Cannot open file '%s'.\n", log_filename);
		return -1;
	}

	fclose(log_f);

	// S_i file
	char Si_filename[1024];
	sprintf(Si_filename, "%sSz.dat", path); 

	FILE *si_f = fopen(Si_filename, "w");
	if (si_f == NULL) {
		errprintf("Cannot open file '%s'.\n", Si_filename);
		return -1;
	}

	int i;
	for (i=meas->num_sites-1; i>=0; i--) {
		fprintf(si_f, "%6.12f\n", meas->Szs[i]);
	}

	fclose(si_f);

	// S_i S_j file
	char SS_filename[1024];
	sprintf(SS_filename, "%sSS.dat", path); 

	FILE *ss_f = fopen(SS_filename, "w");
	if (ss_f == NULL) {
		errprintf("Cannot open file '%s'.\n", SS_filename);
		return -1;
	}

	for (i=meas->num_sites-1; i>=0; i--) {
		fprintf(ss_f, "%6.12f\n", meas->SSs[i]);
	}

	fclose(ss_f);

	return 0;
}
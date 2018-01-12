#include "params.h"
#include "model.h"

int validateParams(const sim_params_t *params) {

	return 0;
}

void printSimParams(FILE *stream, const sim_params_t *params) {

	fprintf(stream, 
			"\n"
			"Heisenberg DMRG\n"
			"******************************\n\n"
			"L = %d         \n"
			"minf = %d      \n"
			"num_sweeps = %d\n"
			"ms = ", params->L, params->minf, params->num_ms);

	for (int i = 0; i < params->num_ms-1; i++) {
		fprintf(stream, "%d, ", params->ms[i]);
	} fprintf(stream, "%d\n", params->ms[params->num_ms-1]);

	double *H_params = params->model->H_params;

	fprintf(stream, 
			"\n"
			"J  = % .4f\n"
			"Jz = % .4f\n"
			"\n"
			"Start Time : %s"
			, H_params[0], H_params[1],
			ctime(params->start_time) );

	if (params->end_time != NULL) {
		fprintf(stream,
			"End   Time : %s"
			, ctime(params->end_time) );
	}

	if (params->runtime > 0) {
		fprintf(stream,
			"CPU Runtime: %.3f seconds\n"
			, params->runtime );
	}

	fprintf(stream,
			"\n"
			"Compiled on git version "VERSION
			"\n"
			"******************************\n\n\n");
}

void freeParams(sim_params_t *params) {

	mkl_free(params->ms);
}
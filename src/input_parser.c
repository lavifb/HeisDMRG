#include "input_parser.h"
#include "model.h"
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int parseInputFile(const char *filename, sim_params_t *params) {

	FILE *fd = fopen(filename, "r");
	if (fd == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	#define MAXLINE 1024
	char line[MAXLINE];
	while (fgets(line, sizeof(line), fd) != NULL) {

		// end line at comment char #
		char *hash_comment = strchr(line, '#');
		if (hash_comment != NULL) {
			*hash_comment = '\0';
		}

		// delimiter characters
		const char *delim = " =:,;\t\r\n";

		// where to read params from
		char *rline = line;

		// check for matrix
		char matline[MAXLINE];
		matline[MAXLINE-1] = '\0';
		
		char *mat_start = strpbrk(line, "{[(");
		if (mat_start != NULL) {
			
			*mat_start = ' ';
			strncpy(matline, line, MAXLINE-1);
			
			char *mat_end = strpbrk(matline, "}])");
			while (mat_end == NULL) {
				fgets(line, sizeof(line), fd);
				if (line == NULL) {
					errprintf("Failed to find closing matrix brace.\n");
					return -1;
				}
				char *hash_comment = strchr(line, '#');
				if (hash_comment != NULL) {
					*hash_comment = '\0';
				}
				strncat(matline, line, MAXLINE-1);
				mat_end = strpbrk(matline, "}])");
			}
			
			*mat_end = '\0';
			rline = matline;
		}

		char *paramName = strtok(rline, delim);
		if (paramName == NULL) {
			continue;
		}

		char *vals[1024];
		int num_vals = 0;

		vals[num_vals] = strtok(NULL, delim);

		while (vals[num_vals]) {
			num_vals++;
			vals[num_vals] = strtok(NULL, delim);
		}

		if (num_vals == 0) {
			continue;
		}

		// Store parameter values
		if (strcmp(paramName, "L") == 0) {
			int L = atoi(vals[0]);

			if (L < 2 || L%2 == 1) {
				errprintf("Parameter 'L' must be even and positive.\n");
				return -2;
			}
			params->L = L;
		} else if (strcmp(paramName, "ms") == 0) {
			int *ms = mkl_malloc(num_vals * sizeof(int), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				ms[i] = atoi(vals[i]);
				if (ms[i] < 1) {
					errprintf("Parameter 'ms' must all be positive.\n");
					return -2;
				}
			}
			params->ms = ms;
			params->num_ms = num_vals;
			 
		} else if (strcmp(paramName, "minf") == 0) {
			int minf = atoi(vals[0]);
			if (minf < 1) {
				errprintf("Parameter 'minf' must be positive.\n");
				return -2;
			}
			params->minf = minf;
		} else if (strcmp(paramName, "d_model") == 0) {
			int d_model = atoi(vals[0]);
			if (d_model < 1) {
				errprintf("Parameter 'd_model' must be positive.\n");
				return -2;
			}
			params->model->d_model = d_model;
		} else if (strcmp(paramName, "H1") == 0) {
			// make sure d_model is declared to check matrix size
			int d_model = params->model->d_model;
			if (d_model < 1) {
				errprintf("Please declare parameter 'd_model' before any matrices.\n");
				return -2;
			}

			if (d_model * d_model != num_vals) {
				errprintf("Parameter 'H1' must be a %d x %d matrix. %d values found.\n", d_model, d_model, num_vals);
				return -2;
			}

			double *H1 = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				H1[i] = atof(vals[i]);
				if (isnan(H1[i])) {
					errprintf("Parameter 'H1' must be all floats.\n");
					return -2;
				}
			}
			params->model->H1 = H1;
		} else if (strcmp(paramName, "Sz") == 0) {
			// make sure d_model is declared to check matrix size
			int d_model = params->model->d_model;
			if (d_model < 1) {
				errprintf("Please declare parameter 'd_model' before any matrices.\n");
				return -2;
			}

			if (d_model * d_model != num_vals) {
				errprintf("Parameter 'Sz' must be a d_model x d_model matrix.\n");
				return -2;
			}

			double *Sz = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				Sz[i] = atof(vals[i]);
				if (isnan(Sz[i])) {
					errprintf("Parameter 'Sz' must be all floats.\n");
					return -2;
				}
			}
			params->model->Sz = Sz;
		} else if (strcmp(paramName, "Sp") == 0) {
			// make sure d_model is declared to check matrix size
			int d_model = params->model->d_model;
			if (d_model < 1) {
				errprintf("Please declare parameter 'd_model' before any matrices.\n");
				return -2;
			}

			if (d_model * d_model != num_vals) {
				errprintf("Parameter 'Sp' must be a d_model x d_model matrix.\n");
				return -2;
			}

			double *Sp = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				Sp[i] = atof(vals[i]);
				if (isnan(Sp[i])) {
					errprintf("Parameter 'Sp' must be all floats.\n");
					return -2;
				}
			}
			params->model->Sp = Sp;
		} else if (strcmp(paramName, "J") == 0) {
			int J = atof(vals[0]);
			params->model->J = J;
		} else if (strcmp(paramName, "Jz") == 0) {
			int Jz = atof(vals[0]);
			params->model->Jz = Jz;
		}
	}

	fclose(fd);

	return 0;
}



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

	int i;
	for (i = 0; i < params->num_ms-1; i++) {
		fprintf(stream, "%d, ", params->ms[i]);
	} fprintf(stream, "%d\n", params->ms[params->num_ms-1]);

	fprintf(stream, 
			"\n"
			"J  = % .4f\n"
			"Jz = % .4f\n"
			"\n"
			"Start Time : %s"
			, params->model->J, params->model->Jz,
			asctime(localtime(params->start_time)));

	if (params->runtime > 0) {
		fprintf(stream,
			"CPU Runtime: %.2f seconds\n"
			, params->runtime );
	}

	fprintf(stream,
			"\n"
			"******************************\n\n");
}

void freeParams(sim_params_t *params) {

	mkl_free(params->ms);
	mkl_free(params->start_time);
}
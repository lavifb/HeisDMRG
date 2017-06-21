#include "input_parser.h"
#include "model.h"
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parseInputFile(const char *filename, sim_params_t *params) {

	FILE *fd = fopen(filename, "r");
	if (fd == NULL) {
		printf("Cannot open file '%s'.\n", filename);
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
					printf("Failed to find closing matrix brace.\n");
					return -1;
				}
				// printf("%s\n", line);
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
				printf("Parameter 'L' must be even and positive.\n");
				return -2;
			}
			params->L = L;
		} else if (strcmp(paramName, "ms") == 0) {
			int *ms = mkl_malloc(num_vals * sizeof(int), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				ms[i] = atoi(vals[i]);
				if (ms[i] < 1) {
					printf("Parameter 'ms' must all be positive.\n");
					return -2;
				}
			}
			params->ms = ms;
			params->num_ms = num_vals;
			 
		} else if (strcmp(paramName, "minf") == 0) {
			int minf = atoi(vals[0]);
			if (minf < 1) {
				printf("Parameter 'minf' must be positive.\n");
				return -2;
			}
			params->minf = minf;
		} else if (strcmp(paramName, "H1") == 0) {
			double *H1 = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				H1[i] = atof(vals[i]);
				if (isnan(H1[i])) {
					printf("Parameter 'H1' must be all floats.\n");
					return -2;
				}
			}
			params->model->H1 = H1;
		} else if (strcmp(paramName, "Sz") == 0) {
			double *Sz = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				Sz[i] = atof(vals[i]);
				if (isnan(Sz[i])) {
					printf("Parameter 'Sz' must be all floats.\n");
					return -2;
				}
			}
			params->model->Sz = Sz;
		} else if (strcmp(paramName, "Sp") == 0) {
			double *Sp = mkl_malloc(num_vals * sizeof(double), MEM_DATA_ALIGN);

			int i;
			for (i = 0; i < num_vals; i++) {
				Sp[i] = atof(vals[i]);
				if (isnan(Sp[i])) {
					printf("Parameter 'Sp' must be all floats.\n");
					return -2;
				}
			}
			params->model->Sp = Sp;
		}


	}

	fclose(fd);

	return 0;
}

void printSimParams(const sim_params_t *params) {

	printf( "L = %d         \n"
			"minf = %d      \n"
			"num_sweeps = %d\n"
			"ms = ", params->L, params->minf, params->num_ms);

	int i;
	for (i = 0; i < params->num_ms-1; i++) {
		printf("%d, ", params->ms[i]);
	} printf("%d\n", params->ms[params->num_ms-1]);

}
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

	char line[1024];
	while (fgets(line, sizeof(line), fd) != NULL) {

		// end line at comment char #
		char *hash_comment = strchr(line, '#');
		if (hash_comment != NULL) {
			*hash_comment = '\0';
		}

		// delimiter characters
		const char *delim = " =:,;\t\r\n";

		char *paramName = strtok(line, delim);
		if (paramName == NULL) {
			continue;
		}

		char *vals[1024];
		int num_vals = 0;

		/*
		// check for matrix
		char *mat_start = strchr(line, '{');
		if (mat_start != NULL) {
			
		}
		*/
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
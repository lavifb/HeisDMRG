#include "model.h"
#include "params.h"
#include "block.h"
#include "hamil.h"
#include "meas.h"
#include "linalg.h"
#include "dmrg.h"
#include "logio.h"
#include "matio.h"
#include "util.h"
#include <mkl.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define USAGE_STATEMENT fprintf(stderr, "usage: tdmrg_test [-sdr] [-m num] [-w num] [-wt num] [-dt float] [-L num] [-c dir_path]\n"); exit(1);

int main(int argc, char *argv[]) {

	int L    = 32;
	int mm   = 20;
	int n_ms = 8;
	int argsave = 0;
	int argrunsave = 0;
	int argrefl = 0;
	char *cont_dir = "";
	int n_ts = 10;
	double dtau = 0.1;

	// Processing command line arguments
	for (int i=1; i<argc; i++) {
		if (strcmp(argv[i], "-m") == 0) {
			if (i+1 < argc) {
				mm = atoi(argv[i+1]);
				if (mm <= 0) {
					errprintf("tdmrg_test: number of tracked states '%s' must be a positive number\n", argv[i+1]);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-m' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (strcmp(argv[i], "-w") == 0) {
			if (i+1 < argc) {
				n_ms = atoi(argv[i+1]);
				if (n_ms <= 0) {
					errprintf("tdmrg_test: number of sweeps '%s' must be a positive number\n", argv[i+1]);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-w' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (strcmp(argv[i], "-wt") == 0) {
			if (i+1 < argc) {
				n_ts = atoi(argv[i+1]);
				if (n_ts < 0) {
					errprintf("tdmrg_test: number of tdmrg sweeps '%s' must be a non-negative number\n", argv[i+1]);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-wt' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (strcmp(argv[i], "-dt") == 0) {
			if (i+1 < argc) {
				dtau = atof(argv[i+1]);
				if (dtau < 0) {
					errprintf("tdmrg_test: time step '%s' must be a non-negative float\n", argv[i+1]);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-dt' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (strcmp(argv[i], "-L") == 0) {
			if (i+1 < argc) {
				L = atoi(argv[i+1]);
				if (L <= 4 || L%2 == 1) {
					errprintf("tdmrg_test: size of system '%s' must be an even number greater than 4\n", argv[i+1]);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-w' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (strcmp(argv[i], "-c") == 0) {
			if (i+1 < argc) {
				cont_dir = argv[i+1];
				argsave = 1;
				argrunsave = 1;

				// check if we have a valid dir
				struct stat sb;
			    if (stat(cont_dir, &sb) < 0 || !S_ISDIR(sb.st_mode)) {
					errprintf("tdmrg_test: saved block path '%s' must be a valid path to a directory\n", cont_dir);
					USAGE_STATEMENT
				}
				i++;
			} else {
				errprintf("tdmrg_test: option '-c' requires an argument\n");
				USAGE_STATEMENT
			}
		} else if (argv[i][0] == '-') {
			for (int j=1; argv[i][j]!='\0'; j++) {
				switch (argv[i][j]) {
					case 's': // save blocks
						argsave = 1;
						argrunsave = 1;
						break;
					case 'd': // save blocks during runtime but delete on completion
						argrunsave = 1;
						break;
					case 'r': // use reflection symmetry
						argrefl = 1;
						break;
					default:
						errprintf("tdmrg_test: illegal option '-%c'\n", argv[i][j]);
						USAGE_STATEMENT
						break;
				}
			}
		}
	}

	// track memory usage
	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	sim_params_t *params = mkl_calloc(sizeof(sim_params_t), 1, MEM_DATA_ALIGN);
	params->L      = L;
	params->minf   = mm;
	params->num_ms = n_ms;
	params->ms     = mkl_malloc(n_ms * sizeof(int), MEM_DATA_ALIGN);
	for (int i = 0; i < n_ms; i++) { params->ms[i] = mm; }

	model_t *model = newHeis2Model();
	model->fullLength = params->L;
	compileParams(model);

	params->model = model;
	params->save_blocks = argrunsave;
	params->reflection = argrefl;

	params->num_ts = n_ts;
	params->dtau = dtau;

	time_t start_time = time(NULL);

	// setup path for continuing run
	if (cont_dir != "") {
		params->continue_run = 1;
		sprintf(params->block_dir, cont_dir);
	}
	// file path for saving blocks dir
	sprintf(params->block_dir, "temp-tL%d_M%d_sim_%ld", params->L, params->ms[params->num_ms-1], start_time);
	mkdir(params->block_dir, 0755);


	printf("Running tdmrg test on version "VERSION".\n\n");

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	meas_data_t *meas = fin_dmrg(params);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double runtime = (t_end.tv_sec - t_start.tv_sec);
	runtime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000000.0;

	printf("M=%d finished in %.3f seconds.\n\n", mm, runtime);


	// Delete temporary files
	if (!argsave) {
		rmrf(params->block_dir);
	}

	int success = 0;

	freeMeas(meas);
	freeModel(model);
	freeParams(params);
	mkl_free(params);

	mkl_free_buffers();
	int nbuffers;
	MKL_INT64 nbytes_alloc, nbytes_alloc_peak;
	nbytes_alloc = mkl_mem_stat(&nbuffers);
	if (nbytes_alloc > 0) {
		warnprintf("MKL reports a memory leak of %lld bytes in %d buffer(s).\n", nbytes_alloc, nbuffers);
		success = -1;
	}

	nbytes_alloc_peak = mkl_peak_mem_usage(MKL_PEAK_MEM);
	printf("Peak memory used is %lld bytes.\n", nbytes_alloc_peak);

	return 0;
}
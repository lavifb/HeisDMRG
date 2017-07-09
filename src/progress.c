#include "progress.h"
#include <stdio.h>

void printProgressBar(int tot) {

	char *pbar = malloc((tot+1) * sizeof(char));

	memset(pbar, '.', tot);
	pbar[tot] = '\0';

	printf("[%s]", pbar);
	fflush(stdout);

	free(pbar);
}

void updateProgressBar(int filled, int tot) {

	char *pbar = malloc((tot+1) * sizeof(char));

	memset(pbar, '.', tot);
	memset(pbar, '#', filled);
	pbar[tot] = '\0';

	printf("\r[%s]", pbar);
	fflush(stdout);

	free(pbar);
}
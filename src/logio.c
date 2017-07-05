#include "logio.h"
#include "block.h"
#include <stdio.h>

FILE *f_log = NULL;

void logBlock(DMRGBlock *block) {

	if (f_log != NULL) {
		fprintf(f_log, "%-5d%- 20.12f%- 20.12e\n",
			block->length, block->energy / block->fullLength, block->trunc_err);
		fflush(f_log);
	}
}

void logSweepEnd() {

	if (f_log != NULL) {
		fprintf(f_log, "\n\n");
	}
}
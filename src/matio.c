#include "linalg.h"
#include "input_parser.h"
#include <stdio.h>

/*  Saves matrix binary data to file.
	Note: file at specified path is overwritten.

	filename: Path where data is saved
	A       : Matrix to be saved
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int saveMat(char *filename, MAT_TYPE *A, int matsize) {

	FILE *m_f = fopen(filename, "wb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fwrite(A, sizeof(MAT_TYPE), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not written properly to file '%s'.\n", filename);
		return -2;
	}
	fclose(m_f);

	return 0;
}

/*  Reads matrix binary data from file.

	filename: Path where data is read
	A       : Matrix to be read
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int readMat(char *filename, MAT_TYPE *A, int matsize) {

	FILE *m_f = fopen(filename, "rb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fread(A, sizeof(MAT_TYPE), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not read properly from file '%s'.\n", filename);
		return -2;
	}
	fclose(m_f);

	return 0;
}
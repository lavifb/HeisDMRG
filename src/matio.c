#include "linalg.h"
#include "util.h"
#include <mkl.h>
#include <stdio.h>

char temp_dir[1024];

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
		errprintf("Matrix not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}
	fclose(m_f);

	return 0;
}

/*  Saves double matrix binary data to file.
	Note: file at specified path is overwritten.

	filename: Path where data is saved
	A       : Matrix to be saved
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int dsaveMat(char *filename, double *A, int matsize) {

	FILE *m_f = fopen(filename, "wb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fwrite(A, sizeof(double), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not written properly to file '%s'.\n", filename);
		return -2;
	}
	fclose(m_f);

	return 0;
}

/*  Reads double matrix binary data from file.

	filename: Path where data is read
	A       : Matrix to be read
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int dreadMat(char *filename, double *A, int matsize) {

	FILE *m_f = fopen(filename, "rb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fread(A, sizeof(double), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}
	fclose(m_f);

	return 0;
}

/*  Saves complex matrix binary data to file.
	Note: file at specified path is overwritten.

	filename: Path where data is saved
	A       : Matrix to be saved
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int zsaveMat(char *filename, MKL_Complex16 *A, int matsize) {

	FILE *m_f = fopen(filename, "wb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fwrite(A, sizeof(MKL_Complex16), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not written properly to file '%s'.\n", filename);
		return -2;
	}
	fclose(m_f);

	return 0;
}

/*  Reads complex matrix binary data from file.

	filename: Path where data is read
	A       : Matrix to be read
	matsize : full size of A. If A is N*N set matsize=N*N

*/
int zreadMat(char *filename, MKL_Complex16 *A, int matsize) {

	FILE *m_f = fopen(filename, "rb");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filename);
		return -1;
	}

	int count = fread(A, sizeof(MKL_Complex16), matsize, m_f);
	if (count != matsize) {
		errprintf("Matrix not read properly from file '%s'. Expected %d items but read %d.\n", filename, matsize, count);
		return -2;
	}
	fclose(m_f);

	return 0;
}

MKL_INT64 getMemStat() {
	int nbuffers;
	MKL_INT64 nbytes_alloc;
	nbytes_alloc = MKL_Mem_Stat(&nbuffers);

	return nbytes_alloc;
} 

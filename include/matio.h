#ifndef MATIO_H
#define MATIO_H

int saveMat(char *filename, MAT_TYPE *A, int matsize);

int readMat(char *filename, MAT_TYPE *A, int matsize);

int dsaveMat(char *filename, double *A, int matsize);

int dreadMat(char *filename, double *A, int matsize);

int zsaveMat(char *filename, MKL_Complex16 *A, int matsize);

int zreadMat(char *filename, MKL_Complex16 *A, int matsize);

#endif
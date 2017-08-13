#ifndef MATIO_H
#define MATIO_H

int saveMat(char *filename, MAT_TYPE *A, int matsize);

int readMat(char *filename, MAT_TYPE *A, int matsize);

#endif
#include "model.h"
#include "block.h"
#include "dmrg.h"
#include <mkl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("Heisenberg DMRG\n\n");

    int L = 100;
    int m = 5;

    ModelParams *model = (ModelParams *)mkl_malloc(sizeof(ModelParams), MEM_DATA_ALIGN);

    #define N 2
    model->dModel = N;
    model->J  = 1;
    model->Jz = 1;

    // One site matrices
    double H1[N*N] = {0.0, 0.0, 0.0,  0.0};
    double Sz[N*N] = {0.5, 0.0, 0.0, -0.5};
    double Sp[N*N] = {0.0, 0.0, 1.0,  0.0};
    double Id[N*N] = {1.0, 0.0, 0.0,  1.0};

    model->H1 = H1;
    model->Sz = Sz;
    model->Sp = Sp;
    model->Id = Id;
    

    DMRGBlock *sys = inf_dmrg(L, m, model);

    freeDMRGBlock(sys);
}
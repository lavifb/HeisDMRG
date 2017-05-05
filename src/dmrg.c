#include "dmrg.h"
#include "block.h"
#include "linalg.h"
#include <mkl.h>
// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>

/* Single DMRG step
   
   m: truncation dimension size
*/
void single_step(DMRGBlock *sys, const DMRGBlock *env, const int m) {

    DMRGBlock *sys_enl, *env_enl;
    ModelParams *model = sys->model;

    sys_enl = enlargeBlock(sys);

    if (sys == env) { // Don't recalculate
        env_enl = sys_enl;
    }
    else {
        env_enl = enlargeBlock(env);
    }

    int dimSys = sys_enl->dBlock;
    int dimEnv = env_enl->dBlock;
    int dimSup = dimSys * dimEnv;

    double *Isys = identity(dimSys);
    double *Ienv = identity(dimEnv);

    // Superblock Hamiltonian
    double *Hs = HeisenH_int(model->J, model->Jz, dimSys, dimEnv, 
                    sys_enl->ops[1], env_enl->ops[2], model->Sz, model->Sp);
    kron(1.0, dimSys, dimEnv, sys_enl->ops[0], Ienv, Hs);
    kron(1.0, dimSys, dimEnv, Isys, env_enl->ops[0], Hs);

    mkl_free(Isys);
    mkl_free(Ienv);

    double *energies = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);

    double *U = (double *)mkl_malloc(dimSup*dimSup * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(U, MEM_DATA_ALIGN);
    memcpy(U, Hs, dimSup*dimSup * sizeof(double));

    int info;

    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', dimSup, U, dimSup, energies);
    if (info > 0) {
        printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
        exit(1);
    }

    double energy = energies[0]; // record ground state energy
    printf("E/L = %f\n", energy / sys_enl->length);

    double *psi0 = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);
    memcpy(psi0, U, dimSup * sizeof(double)); // copy over only first eigenvalue
    
    // Density matrix rho
    double *rho = (double *)mkl_calloc(dimSup*dimSup, sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(rho, MEM_DATA_ALIGN);

    cblas_dger(CblasColMajor, dimSup, dimSup, 1.0, psi0, 1, psi0, 1, rho, dimSup);
    mkl_free(psi0);
    double *lambs = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);

    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', dimSup, rho, dimSup, lambs);
    if (info > 0) {
        printf("Failed to find eigenvalues of density matrix\n");
        exit(1);
    }

    // Transformation Matrix
    int mm = (dimSup < m) ? dimSup : m; // use min(dimSup, m) 
    double *trans = (double *)mkl_malloc(dimSup*mm * sizeof(double), MEM_DATA_ALIGN);
    int mt = dimSup - mm; // number of truncated dimensions
    // copy over only mm biggest eigenvalues
    memcpy(trans, rho+(dimSup*mt) , dimSup*mm * sizeof(double));

    mkl_free(rho);
    
    double truncation_err = 0;
    int i;
    for (i = 0; i < mt; i++) {
        truncation_err += lambs[i];
    }
    printf("Truncation Error: %f\n", truncation_err);
    mkl_free(lambs);

    transformOps(sys_enl->num_ops, dimSup, mm, trans, sys_enl->ops);

    // Copy new enlarged block into sys
    freeDMRGBlock(sys);
    sys = sys_enl;
    printf("\n");

    mkl_free(Hs);
    mkl_free(energies);
    mkl_free(U);
}

/* Infinite DMRG Algorithm
   
   L: Maximum length of system
   m: truncation dimension size
*/
DMRGBlock *inf_dmrg(const int L, const int m, ModelParams *model) {

    int num_ops = 3;

    double **ops = (double **)mkl_malloc(num_ops * sizeof(double), MEM_DATA_ALIGN);

    int N = model->dModel;
    int i;
    for (i = 0; i < num_ops; i++) {
        ops[i] = (double *)mkl_malloc(N*N * sizeof(double), MEM_DATA_ALIGN);
    }

    memcpy(ops[0], model->H1, N*N * sizeof(double)); // H
    memcpy(ops[1], model->Sz, N*N * sizeof(double)); // Sz
    memcpy(ops[2], model->Sp, N*N * sizeof(double)); // Sp

    DMRGBlock *sys = createDMRGBlock(model, num_ops, ops);

    while (2*sys->length < L) {
        printf("L = %d", sys->length * 2 + 2);
        single_step(sys, sys, m);
    }

    return sys;
}
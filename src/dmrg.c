#include "dmrg.h"
#include "block.h"
#include "linalg.h"
#include <mkl.h>
#include <assert.h>
// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>

/* Print nice graphic of the system and environment
*/
void printGraphic(DMRGBlock *sys, DMRGBlock *env) {

    char *sys_g = (char *)malloc((sys->length+1) * sizeof(char));
    char *env_g = (char *)malloc((env->length+1) * sizeof(char));

    memset(sys_g, '=', sys->length);
    memset(env_g, '-', env->length);
    sys_g[sys->length] = '\0';
    env_g[env->length] = '\0';

    if (sys->side == 'L') {
        printf("%s**%s\n", sys_g, env_g);
    } else {
        printf("%s**%s\n", env_g, sys_g);
    }

    free(sys_g);
    free(env_g);
}

/* Single DMRG step
   
   m: truncation dimension size

   returns enlarged system block
*/
DMRGBlock *single_step(DMRGBlock *sys, const DMRGBlock *env, const int m) {

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
                    sys_enl->ops[1], sys_enl->ops[2], env_enl->ops[1], env_enl->ops[2]);
    kron(1.0, dimSys, dimEnv, sys_enl->ops[0], Ienv, Hs);
    kron(1.0, dimSys, dimEnv, Isys, env_enl->ops[0], Hs);

    mkl_free(Isys);
    mkl_free(Ienv);

    __assume_aligned(Hs, MEM_DATA_ALIGN);


    // Find ground state
    double *psi0 = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(psi0, MEM_DATA_ALIGN);
    int info;
    int num_es_found;
    double energies[1];
    int *ifail = (int *)mkl_malloc(dimSup * sizeof(int), MEM_DATA_ALIGN);;
    __assume_aligned(ifail, MEM_DATA_ALIGN);
    info = LAPACKE_dsyevx(LAPACK_COL_MAJOR, 'V', 'I', 'U', dimSup, Hs, dimSup, 0.0, 0.0,
            1, 1, 0.0, &num_es_found, energies, psi0, dimSup, ifail);
    if (info > 0) {
        printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
        exit(1);
    }
    mkl_free(ifail);
    mkl_free(Hs);

    // psi0 needs to be arranged as a dimSup * dimEnv to trace out env
    // Put sys_basis on rows and env_basis on the cols by taking transpose
    mkl_dimatcopy('C', 'T', dimEnv, dimSys, 1.0, psi0, dimEnv, dimSys);

    double energy = energies[0]; // record ground state energy
    printf("E/L = %6.10f\n", energy / (sys_enl->length + env_enl->length));

    // Density matrix rho
    double *rho = (double *)mkl_malloc(dimSys*dimSys * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(rho, MEM_DATA_ALIGN);

    // Trace out Environment to make rho
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, dimSys, dimSys, dimEnv, 1.0, psi0, dimSys, psi0, dimSys, 0.0, rho, dimSys);

    mkl_free(psi0);
    double *lambs = (double *)mkl_malloc(dimSys * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(lambs, MEM_DATA_ALIGN);

    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', dimSys, rho, dimSys, lambs);
    if (info > 0) {
        printf("Failed to find eigenvalues of density matrix\n");
        exit(1);
    }

    // Transformation Matrix
    int mm = (dimSys < m) ? dimSys : m; // use min(dimSys, m) 
    double *trans = (double *)mkl_malloc(dimSys*mm * sizeof(double), MEM_DATA_ALIGN);
    int mt = dimSys - mm; // number of truncated dimensions
    // copy over only mm biggest eigenvalues
    memcpy(trans, rho+(dimSys*mt) , dimSys*mm * sizeof(double));

    mkl_free(rho);
    
    double truncation_err = 1;
    int i;
    for (i = mt; i < dimSys; i++) {
        truncation_err -= lambs[i];
    }
    printf("Truncation Error: %.10e\n", truncation_err);
    mkl_free(lambs);

    // Transform operators into new basis
    transformOps(sys_enl->num_ops, dimSys, mm, trans, sys_enl->ops);
    sys_enl->dBlock = mm; // set block basis size to transformed value
    mkl_free(trans);

    // Free enlarged environment block
    if (sys_enl != env_enl) {
        freeDMRGBlock(env_enl);
    }

    return sys_enl;
}

/* Infinite System DMRG Algorithm
   
   L: Maximum length of system
   m: truncation dimension size
*/
void inf_dmrg(const int L, const int m, ModelParams *model) {

    DMRGBlock *sys = createDMRGBlock(model);

    while (2*sys->length < L) {
        printf("\nL = %d\n", sys->length * 2 + 2);
        DMRGBlock *newSys = single_step(sys, sys, m);
        freeDMRGBlock(sys);
        sys = newSys;
    }

    freeDMRGBlock(sys);
}

/* Finite System DMRG Algorithm
   
   L         : Length of universe
   m_inf     : truncation dimension size for infinite algorithm for building system
   num_sweeps: number of finite system sweeps
   ms        : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
void fin_dmrg(const int L, const int m_inf, const int num_sweeps, int *ms, ModelParams *model) {
    assert(L%2 == 0);

    DMRGBlock **saved_blocksL = (DMRGBlock **)mkl_calloc((L-4), sizeof(DMRGBlock *), MEM_DATA_ALIGN);
    DMRGBlock **saved_blocksR = (DMRGBlock **)mkl_calloc((L-4), sizeof(DMRGBlock *), MEM_DATA_ALIGN);

    DMRGBlock *sys   = createDMRGBlock(model);

    // Note: saved_blocksL[i] has length i+1
    saved_blocksL[0] = copyDMRGBlock(sys);
    saved_blocksR[0] = copyDMRGBlock(sys);
    saved_blocksR[0]->side = 'R';

    // run infinite algorithm to build up system
    while (2*sys->length < L) {
        printGraphic(sys, sys);
        DMRGBlock *newSys = single_step(sys, sys, m_inf);
        freeDMRGBlock(sys);
        sys = newSys;

        saved_blocksL[sys->length-1] = copyDMRGBlock(sys);
        saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
        saved_blocksR[sys->length-1]->side = 'R';
    }

    // Finite Sweeps
    DMRGBlock *env = copyDMRGBlock(sys);
    int i;
    for (i = 0; i < num_sweeps; i++) {
        int m = ms[i];

        while (1) {
            freeDMRGBlock(env);

            switch (sys->side) {
                case 'L':
                    env = copyDMRGBlock(saved_blocksR[L - sys->length - 3]);
                    break;

                case 'R':
                    env = copyDMRGBlock(saved_blocksL[L - sys->length - 3]);
                    break;
            }

            // Switch sides if at the end of the chain
            if (env->length == 1) {
                DMRGBlock *tempBlock = sys;
                sys = env;
                env = tempBlock;
            }

            printGraphic(sys, env);
            DMRGBlock *newSys = single_step(sys, env, m);
            freeDMRGBlock(sys);
            sys = newSys;

            // Save new block
            switch (sys->side) {
                case 'L':
                    if (saved_blocksL[sys->length-1]) { freeDMRGBlock(saved_blocksL[sys->length-1]); }
                    saved_blocksL[sys->length-1] = copyDMRGBlock(sys);
                    break;

                case 'R':
                    if (saved_blocksR[sys->length-1]) { freeDMRGBlock(saved_blocksR[sys->length-1]); }
                    saved_blocksR[sys->length-1] = copyDMRGBlock(sys);
                    break;
            }

            // Check if sweep is done
            if (sys->side == 'L' && 2 * sys->length == L) {
                break;
            }
        }
    }

    for (i = 0; i < L-4; i++) {
        if (saved_blocksL[i]) { freeDMRGBlock(saved_blocksL[i]); }
        if (saved_blocksR[i]) { freeDMRGBlock(saved_blocksR[i]); }
    }
    mkl_free(saved_blocksL);
    mkl_free(saved_blocksR);

    freeDMRGBlock(env);
    freeDMRGBlock(sys);
}
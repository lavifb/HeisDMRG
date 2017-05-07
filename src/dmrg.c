#include "dmrg.h"
#include "block.h"
#include "linalg.h"
#include <mkl.h>
#include <assert.h>
// #include <stdio.h>
// #include <string.h>
// #include <stdlib.h>

/* Single DMRG step
   
   m: truncation dimension size

   returns enlarged system block
*/
DMRGBlock *single_step(DMRGBlock *sys, DMRGBlock *env, const int m) {

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

    double *energies = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);

    double *U = (double *)mkl_malloc(dimSup*dimSup * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(U, MEM_DATA_ALIGN);
    memcpy(U, Hs, dimSup*dimSup * sizeof(double));

    mkl_free(Hs);

    int info;
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', dimSup, U, dimSup, energies);
    if (info > 0) {
        printf("Failed to find eigenvalues of Superblock Hamiltonian\n");
        exit(1);
    }

    double energy = energies[0]; // record ground state energy
    mkl_free(energies);
    printf("E/L = %6.16f\n", energy / (2 * sys_enl->length));

    double *psi0 = (double *)mkl_malloc(dimSup * sizeof(double), MEM_DATA_ALIGN);
    memcpy(psi0, U, dimSup * sizeof(double)); // copy over only first eigenvalue
    mkl_free(U);

    // Density matrix rho
    double *rho = (double *)mkl_malloc(dimSys*dimSys * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(rho, MEM_DATA_ALIGN);

    // Trace out Environment to make rho
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, dimSys, dimSys, dimEnv, 1.0, psi0, dimSys, psi0, dimSys, 0.0, rho, dimSys);
    mkl_free(psi0);
    double *lambs = (double *)mkl_malloc(dimSys * sizeof(double), MEM_DATA_ALIGN);

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
DMRGBlock *inf_dmrg(const int L, const int m, ModelParams *model) {

    DMRGBlock *sys = createDMRGBlock(model);

    while (2*sys->length < L) {
        printf("\nL = %d\n", sys->length * 2 + 2);
        DMRGBlock *newSys = single_step(sys, sys, m);
        freeDMRGBlock(sys);
        sys = newSys;
    }

    return sys;
}

/* Finite System DMRG Algorithm
   
   L         : Maximum length of system
   m_inf     : truncation dimension size for infinite algorithm for building system
   num_sweeps: number of finite system sweeps
   ms        : list of truncation sizes for the finite sweeps (size num_sweeps)
*/
// DMRGBlock *fin_dmrg(const int L, const int m_inf, const int num_sweeps, int *ms, ModelParams *model) {
//     assert(L%2 == 0);

//     DMRGBlock **saved_blocksL = (DMRGBlock **)mkl_malloc(L*sizeof(DMRGBlock *), MEM_DATA_ALIGN);
//     DMRGBlock **saved_blocksR = (DMRGBlock **)mkl_malloc(L*sizeof(DMRGBlock *), MEM_DATA_ALIGN);

//     return 
// }
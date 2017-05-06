#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>


DMRGBlock *createDMRGBlock(ModelParams *model, const int num_ops, double **ops) {
    DMRGBlock *block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

    block->length = 1;
    block->dBlock = model->dModel;
    block->num_ops = num_ops;
    block->ops = ops;
    block->model  = model;

    return block;
}

void freeDMRGBlock(DMRGBlock *block) {
    if (block) { // check that pointer is actually pointing to something
        freeDMRGBlockOps(block);
        mkl_free(block);
    }
}

void freeDMRGBlockOps(DMRGBlock *block) {
    if (block || block->ops) { // check that pointer is actually pointing to something
        int i;
        for (i=0; i<block->num_ops; i++) {
            mkl_free(block->ops[i]);
        }

        mkl_free(block->ops);
    }
}

void printDMRGBlock(const char *desc, DMRGBlock *block) {
    printf("\n----------\n %s\n", desc);

    printf("length: %d\n", block->length);
    printf("dBlock: %d\n", block->dBlock);
    printf("num_ops: %d\n", block->num_ops);

    print_matrix("H", block->dBlock, block->dBlock, block->ops[0], block->dBlock);
    print_matrix("conn_Sz", block->dBlock, block->dBlock, block->ops[1], block->dBlock);
    print_matrix("conn_Sp", block->dBlock, block->dBlock, block->ops[2], block->dBlock);

    printf("\n");
}

// TODO: use same chunk of memory every time to not reallocate on every enlargement
DMRGBlock *enlargeBlock(const DMRGBlock *block) {

    DMRGBlock *enl_block = (DMRGBlock *)MKL_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
    enl_block->length  = block->length + 1;
    enl_block->dBlock  = block->dBlock * block->model->dModel;
    enl_block->num_ops = block->num_ops;
    enl_block->model   = block->model;

    enl_block->ops = enlargeOps(block);

    return enl_block;
}

/*  Operator Dictionary:
    0: H
    1: conn_Sz
    2: conn_Sp
    
    ----------

    H_enl = kron(H, I_d) + kron(I_m, H1) + H_int(conn_Sz, conn_Sp, Sz, Sp)
    conn_Sz = kron(I_m, Sz)
    conn_Sp = kron(I_m, Sp) 
*/

double **enlargeOps(const DMRGBlock *block) {
    double **enl_ops = (double **)mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);

    ModelParams *model = block->model;
    int dModel  = model->dModel;
    int dim     = block->dBlock;
    int enl_dim = dModel * dim;

    double *I_m = identity(dim);
    // print_matrix("I_m", dim, dim, I_m, dim);

    // print_matrix("Sp1", dModel, dModel, model->Sp, dModel);
    // print_matrix("Sz1", dModel, dModel, model->Sz, dModel);
    // print_matrix("H1 ", dModel, dModel, model->H1, dModel);

    // H_enl
    printf("Building H_enl\n");
    enl_ops[0] = HeisenH_int(model->J, model->Jz, dim, dModel, 
                    block->ops[1], block->ops[2], model->Sz, model->Sp);
    // print_matrix("H_int", enl_dim, enl_dim, enl_ops[0], enl_dim);
    // printf("Taking krons for H_enl\n");
    kron(1.0, dim, dModel, block->ops[0], model->Id, enl_ops[0]);
    kron(1.0, dim, dModel, I_m, model->H1, enl_ops[0]);
    // printf("Built H_enl\n");


    // conn_Sz
    enl_ops[1] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sz, enl_ops[1]);
    // print_matrix("conn_Sz", enl_dim, enl_dim, enl_ops[1], enl_dim);

    // conn_Sp
    enl_ops[2] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sp, enl_ops[2]);
    // print_matrix("conn_Sp", enl_dim, enl_dim, enl_ops[2], enl_dim);
    // printf("Built op_enl\n");

    mkl_free(I_m);
    return enl_ops;
}

/*  Transform an entire set of operators at once.
*/
void transformOps(const int numOps, const int opDim, const int newDim, const double *restrict trans, double **ops) {

    double *newOp = (double *)mkl_malloc(newDim*newDim * sizeof(double), MEM_DATA_ALIGN);
    double *temp  = (double *)mkl_malloc(newDim*opDim  * sizeof(double), MEM_DATA_ALIGN);
    __assume_aligned(trans, MEM_DATA_ALIGN);
    __assume_aligned(newOp, MEM_DATA_ALIGN);
    __assume_aligned(temp , MEM_DATA_ALIGN);

    int i;
    for (i = 0; i < numOps; i++) {
        __assume_aligned(ops[i], MEM_DATA_ALIGN);
        cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, newDim, opDim , opDim, 1.0, trans, opDim, ops[i], opDim, 0.0, temp, newDim);
        cblas_dgemm(CblasColMajor, CblasNoTrans  , CblasNoTrans, newDim, newDim, opDim, 1.0, temp, newDim, trans, opDim, 0.0, newOp, newDim);
        ops[i] = (double *)mkl_realloc(ops[i], newDim*newDim * sizeof(double));
        memcpy(ops[i], newOp, newDim*newDim * sizeof(double)); // copy newOp back into ops[i]
    }

    mkl_free(temp);
    mkl_free(newOp);
}
#include "block.h"
#include "model.h"
#include "linalg.h"
#include <mkl.h>


DMRGBlock *createDMRGBlock(ModelParams *model) {
    DMRGBlock *block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

    block->length = 1;
    block->side = 'L';

    int dim = model->dModel;
    block->dBlock = dim;
    block->num_ops = model->num_ops;
    block->model  = model;

    // copy operators
    block->ops = (double **)mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);
    int i;
    for (i = 0; i < block->num_ops; i++) {
        block->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
        memcpy(block->ops[i], model->initOps[i], dim*dim * sizeof(double));
    }

    return block;
}

DMRGBlock *copyDMRGBlock(DMRGBlock *orig) {
    DMRGBlock *newBlock = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);

    newBlock->length  = orig->length;
    newBlock->side  = orig->side;
    int dim = orig->dBlock;
    newBlock->dBlock  = dim;
    newBlock->num_ops = orig->num_ops;
    newBlock->model   = orig->model;

    // Copy all matrices (not just pointers)
    newBlock->ops = (double **)mkl_malloc(newBlock->num_ops * sizeof(double *), MEM_DATA_ALIGN);
    int i;
    for (i = 0; i < newBlock->num_ops; i++) {
        newBlock->ops[i] = (double *)mkl_malloc(dim*dim * sizeof(double), MEM_DATA_ALIGN);
        memcpy(newBlock->ops[i], orig->ops[i], dim*dim * sizeof(double));
    }
    
    return newBlock;
}

void freeDMRGBlock(DMRGBlock *block) {
    freeDMRGBlockOps(block);
    mkl_free(block);
}

void freeDMRGBlockOps(DMRGBlock *block) {
    int i;
    for (i=0; i<block->num_ops; i++) {
        mkl_free(block->ops[i]);
    }

    mkl_free(block->ops);
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

DMRGBlock *enlargeBlock(const DMRGBlock *block) {

    DMRGBlock *enl_block = (DMRGBlock *)mkl_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
    enl_block->length  = block->length + 1;
    enl_block->dBlock  = block->dBlock * block->model->dModel;
    enl_block->num_ops = block->num_ops;
    enl_block->model   = block->model;
    enl_block->side    = block->side;

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

    // H_enl
    enl_ops[0] = HeisenH_int(model->J, model->Jz, dim, dModel, 
                    block->ops[1], block->ops[2], model->Sz, model->Sp);
    kron(1.0, dim, dModel, block->ops[0], model->Id, enl_ops[0]);
    kron(1.0, dim, dModel, I_m, model->H1, enl_ops[0]);

    // conn_Sz
    enl_ops[1] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sz, enl_ops[1]);

    // conn_Sp
    enl_ops[2] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sp, enl_ops[2]);

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
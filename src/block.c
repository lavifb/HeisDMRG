#include "block.h"
#include "linalg.h"


// void createBlock(DMRGBlock *block, int length, int basis_size) {
//     block
// }

void freeDMRGBlock(DMRGBlock *block) {
    for (int i=0; i<block->num_ops; i++) {
        mkl_free(block->ops[i])
    }

    mkl_free(block->ops);
    mkl_free(block);
}

// TODO: use same chunk of memory every time to not reallocate on every enlargement
DMRGBlock *enlargeBlock(const DMRGBlock *block) {

    DMRGBlock *enl_block = (DMRGBlock *)MKL_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
    enl_block->length = block->length + 1;
    enl_block->basis_size = block->basis_size * block->model->dModel;
    enl_block->num_ops = block->num_ops;
    enl_block->model = block->model;

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
    double **enl_ops = mkl_malloc(block->num_ops * sizeof(double *), MEM_DATA_ALIGN);

    ModelParams *model = block->model;
    int dModel  = model->dModel;
    int dim     = block->basis_size;
    int enl_dim = dModel * dim;

    int I_m = identiy(dim);

    // H_enl
    enl_ops[0] = HeisenH_int(block->J, block->Jz, dim, dModel, 
                    block->ops[1], block->ops[2], model->Sz, model->Sp);
    kron(1.0, dim, dModel, block->ops[0], model->Id, enl_ops[0]);
    kron(1.0, dim, dModel, I_m, model->H1, enl_ops[0]);

    // conn_Sz
    enl_ops[1] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sz, enl_ops[1]);

    // conn_Sp
    enl_ops[2] = (double *)mkl_calloc(enl_dim*enl_dim, sizeof(double), MEM_DATA_ALIGN);
    kron(1.0, dim, dModel, I_m, model->Sp, enl_ops[1]);

    mkl_free(I_m);
    return enl_ops;
}
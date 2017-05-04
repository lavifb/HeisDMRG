#include "block.h"


// void createBlock(DMRGBlock *block, int length, int basis_size) {
// 	block
// }

// TODO: use same chunk of memory every time to not reallocate on every enlargement
DMRGBlock *enlargeBlock(DMRGBlock *block, ) {

	DMRGBlock *enl_block = (DMRGBlock *)MKL_malloc(sizeof(DMRGBlock), MEM_DATA_ALIGN);
	enl_block->length = block->length + 1;
	enl_block->basis_size = block->basis_size + 1;
	enl_block->num_ops = block->num_ops;
	enl_block->model = block->model;

	enl_block->ops = enlargeOps(block)

	return enl_block;
}

/*  Operator Dictionary:
	0: H
	1: conn_Sz
	2: conn_Sp
*/

/*
	H_enl = kron(H, I_d) + kron(I_m, H1) + H_int
	conn_Sz = kron(I_m, Sz)
	conn_Sp = kron(I_m, Sp)	
*/

double **enlargeOps(DMRGBlock *block) {

}

// Interaction part of Heisenberg Hamiltonian
double *Heisen_int(Sz1, Sp1, Sz2, Sp2) {
	
}
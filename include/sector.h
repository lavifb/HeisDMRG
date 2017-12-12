#ifndef SECTOR_H
#define SECTOR_H

#include "block.h"
#include "uthash.h"

#define HASH_IND_MIN_SIZE 32

typedef struct {
	int id;           	// qn storing indexes key for uthash.
	int num_ind;      	// number of stored qns
	int *inds;        	// indexes with quantum number id
	int inds_size;    	// current allocation size of inds
	UT_hash_handle hh;	// makes this structure hashable
} sector_t;

sector_t *createSector(const int id);

void sectorPush(sector_t *sec, const int i);

void freeSector(sector_t *sec);

sector_t *sectorize(const DMRGBlock *block);

void freeSectors(sector_t *sectors);

sector_t *getRestrictedBasis(sector_t *const sys_enl_sectors, sector_t *const env_enl_sectors, const int target_mz, const int dimEnv, int *num_restr_indp, int *restr_basis_inds);

#endif
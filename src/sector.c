#include "sector.h"
#include "block.h"
#include "uthash.h"
#include <mkl.h>


/*	Create sector
*/
sector_t *createSector(const int id) {

	sector_t *sec = (sector_t *)mkl_malloc(sizeof(sector_t), MEM_DATA_ALIGN);
	sec->id = id;
	sec->num_ind = 0;
	sec->inds_size = HASH_IND_MIN_SIZE;
	sec->inds = (int *)mkl_malloc(HASH_IND_MIN_SIZE * sizeof(int), MEM_DATA_ALIGN);

	return sec;
}

/* Push index onto sector
*/
void sectorPush(sector_t *sec, const int i) {

	if(sec->num_ind >= sec->inds_size) {
		sec->inds_size *= 2;
		sec->inds = mkl_realloc(sec->inds, sec->inds_size * sizeof(int));
	}

	sec->inds[sec->num_ind] = i;
	sec->num_ind++;
}

void freeSector(sector_t *sec) {

	mkl_free(sec->inds);
	mkl_free(sec);
}

/*	Create hashtable to track indexes of basis vectors corresponding to qns (called a sector)
*/
sector_t *sectorize(const DMRGBlock *block) {

	sector_t *secs = NULL; // Initialize uthash

	for (int i = 0; i < block->d_block; i++) {
		int id = block->mzs[i];
		sector_t *qn_sec;

		HASH_FIND_INT(secs, &id, qn_sec);

		if (qn_sec == NULL) { // create new entry in hashtable
			qn_sec = createSector(id);
			sectorPush(qn_sec, i);
			HASH_ADD_INT(secs, id, qn_sec);
		} else { // add to entry
			sectorPush(qn_sec, i);
		}
	}

	return secs;
}

/*	free sector hashtable
*/
void freeSectors(sector_t *sectors) {

	sector_t *sec, *tmp;

	HASH_ITER(hh, sectors, sec, tmp) {
		HASH_DEL(sectors, sec);		// delete from hash
		freeSector(sec);       		// free sector
	}
}

/*  Get the desired retricted basis based on the target_mz.

	sys_enl_sectors  : System block sectors
	env_enl_sectors  : Environment block sectors
	target_mz        : Sector to target
	dimEnv 	         : Dimension of enlarged environment block. Used to get restricted basis inds

	num_restr_indp   : Pointer of size 1 to store number of restricted basis inds
	restr_basis_inds : Pointer of size dimEnv*dimSys to store restricted basis inds

	returns relevant sectors for the super block.
*/
sector_t *getRestrictedBasis(sector_t *const sys_enl_sectors, sector_t *const env_enl_sectors, const int target_mz, const int dimEnv, 
								int *num_restr_indp, int *restr_basis_inds) {

	// Create sectors for use in the super-block
	sector_t *sup_sectors = NULL;

	// indexes used for restricting Hs
	int num_restr_ind = 0;

	// loop over sys_enl_sectors and find only desired indexes
	for (sector_t *sys_enl_sec=sys_enl_sectors; sys_enl_sec != NULL; sys_enl_sec=sys_enl_sec->hh.next) {

		int sys_mz = sys_enl_sec->id;

		sector_t *sup_sec;
		sup_sec = createSector(sys_mz);
		HASH_ADD_INT(sup_sectors, id, sup_sec);

		int env_mz = target_mz - sys_mz;

		// pick out env_enl_sector with mz = env_mz
		sector_t *env_enl_sec;
		HASH_FIND_INT(env_enl_sectors, &env_mz, env_enl_sec);
		if (env_enl_sec != NULL) {
			int i, j;
			for (i = 0; i < sys_enl_sec->num_ind; i++) {
				for (j = 0; j < env_enl_sec->num_ind; j++) {
					// save restricted index and save into sup_sectors
					sectorPush(sup_sec, num_restr_ind);
					restr_basis_inds[num_restr_ind] = sys_enl_sec->inds[i]*dimEnv + env_enl_sec->inds[j];
					num_restr_ind++;
				}
			}
		}
	}

	*num_restr_indp = num_restr_ind;
	return sup_sectors;
}
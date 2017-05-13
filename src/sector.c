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

	int j;
	for (j = 0; j < block->d_block; j++) {
		int id = block->mzs[j];
		sector_t *qn_sec;

		HASH_FIND_INT(secs, &id, qn_sec);

		if (qn_sec == NULL) { // create new entry in hashtable
			qn_sec = createSector(id);
			sectorPush(qn_sec, j);
			HASH_ADD_INT(secs, id, qn_sec);
		} else { // add to entry
			sectorPush(qn_sec, j);
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
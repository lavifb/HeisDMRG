#include "util.h"
#include <stdio.h>
#include <ftw.h>


int remove_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
	int rv = remove(fpath);
	if (rv) {
		errprintf("Error removing file at path '%s'", fpath);
	}

	return rv;
}

/*  Function for removing directories with all of their contents.
	Similar to "rm -rf 'path'"
*/
int rmrf(char *path) {
	return nftw(path, remove_cb, 2, FTW_DEPTH | FTW_PHYS);
}
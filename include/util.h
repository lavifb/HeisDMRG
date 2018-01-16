#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

// ANSI colors for print output
#define TERM_RED     "\x1b[31m"
#define TERM_GREEN   "\x1b[32m"
#define TERM_YELLOW  "\x1b[33m"
#define TERM_RESET   "\x1b[0m"

#define errprintf(M, ...) fprintf(stderr, TERM_RED "dmrg: [ERROR] " M TERM_RESET , ##__VA_ARGS__)
#define warnprintf(M, ...) fprintf(stderr, TERM_YELLOW "dmrg: [WARNING] " M TERM_RESET , ##__VA_ARGS__)

#define failprintf(M, ...) fprintf(stdout, TERM_RED M TERM_RESET , ##__VA_ARGS__)
#define passprintf(M, ...) fprintf(stdout, TERM_GREEN M TERM_RESET , ##__VA_ARGS__)

int rmrf(char *path);

#endif
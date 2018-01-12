#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>

// ANSI colors for print output
#define TERM_RED     "\x1b[31m"
#define TERM_GREEN   "\x1b[32m"
#define TERM_RESET   "\x1b[0m"

#define errprintf(M, ...) printf( TERM_RED "[ERROR] " M TERM_RESET , ##__VA_ARGS__)

#endif
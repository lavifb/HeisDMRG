#ifndef LOGIO_H
#define LOGIO_H

#include "block.h"
#include <stdio.h>

extern FILE *f_log;

void logBlock(const DMRGBlock *block);

void logSweepEnd();

#endif
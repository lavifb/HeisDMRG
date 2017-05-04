
# use Intel compiler
CC = icc

# compiler options
CCOPTS = -Wall -O2 -xHost -restrict -DNDEBUG
# CCOPTS = -Wall -O2 -xHost -restrict -DNDEBUG -DMEM_DATA_ALIGN=64

# MKL Library
MKL = -mkl=sequential

BIN := bin/
SRC := src/
INC := include/


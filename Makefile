
# use Intel compiler
CC = icc

# compiler options
CCOPTS  = -Wall -xHost -restrict -DMEM_DATA_ALIGN=64
CCOPTSR = ${CCOPTS} -DNDEBUG -O2
CCOPTSD = ${CCOPTS} -g -O0 -DMKL_DISABLE_FAST_MM=1

# MKL Library
MKL = -mkl=sequential

BIN := bin
SRC := src
INC := include
ODIR:= odir
TEST:= test

# SRC_FILES = $(patsubst $(PUG)/%.pug, $(TEST)/%.html, $(wildcard $(PUG)/[^_]*.pug))

build: proj_main

clean: 
	-rm ${BIN}/*
	-rm ${ODIR}/*
	-rm ${TEST}/*

clean-test:
	-rm ${TEST}/*

src : $(patsubst $(SRC)/%.c, $(ODIR)/%.o, $(wildcard $(SRC)/[^_]*.c))
srcD: $(patsubst $(SRC)/%.c, $(TEST)/%.o, $(wildcard $(SRC)/[^_]*.c))

${ODIR}/%.o: ${SRC}/%.c
	${CC} -c -I${INC}/ ${CCOPTSR} ${MKL} $< -o $@

${TEST}/%.o: ${SRC}/%.c
	${CC} -c -I${INC}/ ${CCOPTSD} ${MKL} $< -o $@

proj_main: src
	 ${CC} -I${INC}/ ${CCOPTSR} ${MKL} -o ${BIN}/dmrg ${ODIR}/*

debug: clean-test srcD
	 ${CC} -I${INC}/ ${CCOPTSD} ${MKL} -o ${TEST}/dmrg_debug ${TEST}/*

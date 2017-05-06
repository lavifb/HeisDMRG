
# use Intel compiler
CC = icc

# compiler options
CCOPTS = -Wall -O2 -xHost -restrict -DNDEBUG -DMEM_DATA_ALIGN=64
# CCOPTS = -Wall -O2 -xHost -g -restrict -DNDEBUG -DMEM_DATA_ALIGN=64

# MKL Library
MKL = -mkl=sequential

BIN := bin
SRC := src
INC := include
ODIR:= odir

# SRC_FILES = $(patsubst $(PUG)/%.pug, $(TEST)/%.html, $(wildcard $(PUG)/[^_]*.pug))

clean: 
	-rm ${BIN}/*
	-rm ${ODIR}/*

src: $(patsubst $(SRC)/%.c, $(ODIR)/%.o, $(wildcard $(SRC)/[^_]*.c))

${ODIR}/%.o: ${SRC}/%.c
	${CC} -c -I${INC}/ ${CCOPTS} ${MKL} $< -o $@

proj_main: src
	 ${CC} -I${INC}/ ${CCOPTS} ${MKL} -o ${BIN}/dmrg ${ODIR}/*
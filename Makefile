
# use Intel compiler
CC = icc

# compiler options
CCOPTS = -Wall -O2 -xHost -restrict -DNDEBUG -DMEM_DATA_ALIGN=64

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

src: $(patsubst $(INC)/%.h, $(ODIR)/%.o, $(wildcard $(INC)/[^_]*.h))

${ODIR}/%.o: ${SRC}/%.c
	${CC} -c -I${INC}/ ${CCOPTS} ${MKL} $< -o $@
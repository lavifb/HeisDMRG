
# use Intel compiler
CC = icc

BIN := bin
SRC := src
INC := include
OBJ := obj
TEST:= test
DBUG:= debug

# compiler options
# CCOPTS  = -Wall -xHost -restrict -std=c99 -DMEM_DATA_ALIGN=64 -DCOMPLEX
CCOPTS  = -Wall -xHost -restrict -std=c99 -DMEM_DATA_ALIGN=64 
CCOPTSR = ${CCOPTS} -DNDEBUG -O2
# CCOPTSR = ${CCOPTS} -DNDEBUG -pg -O2
CCOPTSD = ${CCOPTS} -g -O0 -DMKL_DISABLE_FAST_MM=1

# MKL Library
MKL = -mkl=sequential
# MKL = -mkl=parallel

# Set directory containing PRIMME library
PRIMMEDIR = ../../Repos/primme

LIB = ${MKL} -lprimme -L${PRIMMEDIR}/lib/
INCDIRS = -I${INC}/ -I${PRIMMEDIR}/include/


# SRC_FILES = $(patsubst $(PUG)/%.pug, $(TEST)/%.html, $(wildcard $(PUG)/[^_]*.pug))

build: dirs proj_main test

dirs:
	@mkdir -p odir
	@mkdir -p output
	@mkdir -p debug

clean: 
	-rm -rf ${BIN}/*
	-rm ${OBJ}/*

clean-debug:
	-rm -rf ${DBUG}/*

clean-all: clean clean-debug

clean-main:
	-rm -f ${OBJ}/main.o

src : $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(wildcard $(SRC)/[^_]*.c))
srcD: $(patsubst $(SRC)/%.c, $(DBUG)/%.o, $(wildcard $(SRC)/[^_]*.c))

test: $(patsubst $(TEST)/%.c, $(BIN)/%, $(wildcard $(TEST)/[^_]*.c))

${OBJ}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSR} ${LIB} $< -o $@

${DBUG}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSD} ${LIB} $< -o $@

proj_main: src
	 ${CC} ${INCDIRS} ${CCOPTSR} ${LIB} -o ${BIN}/dmrg ${OBJ}/*

debug: clean-debug srcD
	 ${CC} ${INCDIRS} ${CCOPTSD} ${LIB} -o ${BIN}/dmrg_debug ${DBUG}/*
	 -rm -f ${DBUG}/main.o
	 ${CC} ${INCDIRS} ${CCOPTSD} ${LIB} -o ${BIN}/quick_test_debug ${DBUG}/* ${TEST}/quick_test.c
	 
${BIN}/%: ${TEST}/%.c src clean-main
	${CC} ${INCDIRS} ${CCOPTSR} ${LIB} -o $@ $< ${OBJ}/*

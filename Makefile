
# use Intel compiler
CC = icc

BIN := bin
SRC := src
INC := include
OBJ := obj
TEST:= test
DBUG:= debug

# Set directory containing PRIMME library
# Comment out this line if do not have the PRIMME library
PRIMMEDIR = ../../Repos/primme

# Comment out this line if you want a purely real calculation
# This gives a reasonable performance boost for real calcualtions
#COMPLEX = true

# compiler options
CCOPTS  = -Wall -xHost -restrict -std=c99 -DMEM_DATA_ALIGN=64 -DVERSION=\"$(shell git describe --always)\"

# MKL Library
MKL = -mkl=sequential
# MKL = -mkl=parallel

LIB = ${MKL}
INCDIRS = -I${INC}/

ifdef COMPLEX
	CCOPTS += -DCOMPLEX
endif

ifdef PRIMMEDIR
	CCOPTS += -DUSE_PRIMME
	LIB += -lprimme -L${PRIMMEDIR}/lib/
	INCDIRS += -I${PRIMMEDIR}/include/
endif

CCOPTSR = ${CCOPTS} -DNDEBUG -O2
# CCOPTSR = ${CCOPTS} -DNDEBUG -pg -O2
CCOPTSD = ${CCOPTS} -g -O0 -DMKL_DISABLE_FAST_MM=1


srcs  = $(filter-out ${SRC}/main.c, $(wildcard ${SRC}/*.c))
objs  = $(patsubst ${SRC}/%.c, ${OBJ}/%.o,  ${srcs})
objsD = $(patsubst ${SRC}/%.c, ${DBUG}/%.o, ${srcs})

.PHONY: build
build: proj_main tests

.PHONY: proj_main
proj_main: ${BIN}/dmrg

.PHONY: debug
debug: clean-debug ${BIN}/dmrg_debug ${BIN}/quick_test_debug

.PHONY: clean
clean: 
	-rm -rf ${BIN}/*
	-rm ${OBJ}/*

.PHONY: clean-debug
clean-debug:
	-rm -rf ${DBUG}/*

.PHONY: clean-all
clean-all: clean clean-debug

.PHONY: tests
tests: $(patsubst $(TEST)/%.c, $(BIN)/%, $(wildcard $(TEST)/*.c))

${OBJ}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSR} ${MKL} $< -o $@

${DBUG}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSD} ${MKL} $< -o $@

${BIN}/dmrg: ${SRC}/main.c ${objs}
	${CC} ${INCDIRS} ${CCOPTSR} -o ${BIN}/dmrg ${OBJ}/* ${SRC}/main.c ${LIB}

${BIN}/dmrg_debug: ${SRC}/main.c ${objsD}
	${CC} ${INCDIRS} ${CCOPTSD} -o ${BIN}/dmrg_debug ${DBUG}/* ${SRC}/main.c ${LIB}

${BIN}/quick_test_debug: ${SRC}/quick_test.c ${objsD}
	${CC} ${INCDIRS} ${CCOPTSD} -o ${BIN}/quick_test_debug ${DBUG}/* ${TEST}/quick_test.c ${LIB}
	 
${BIN}/%: ${TEST}/%.c ${objs}
	${CC} ${INCDIRS} ${CCOPTSR} -o $@ $< ${OBJ}/* ${LIB}

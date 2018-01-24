
# use Intel compiler
CC = icc

BIN := bin
SRC := src
INC := include
OBJ := obj
TEST:= test
DBUG:= debug

# Set directory containing PRIMME library
# Comment out this line if do not have the PRIMME library (The code will be WAY slower and may be broken)
PRIMMEDIR = ../primme


# compiler options
CCOPTS  = -Wall -xHost -restrict -std=gnu99 -DMEM_DATA_ALIGN=64 -DVERSION=\"$(shell git describe --always)\"
# add openmp for some parallelization improvements
CCOPTS += -qopenmp

# MKL Library
# MKL = -mkl=sequential
MKL = -mkl=parallel
# MKL = -mkl=cluster

LIB = ${MKL}
INCDIRS = -I${INC}/


ifdef PRIMMEDIR
	LIB += -lprimme -L${PRIMMEDIR}/lib/
	INCDIRS += -I${PRIMMEDIR}/include/
endif

CCOPTSR = ${CCOPTS} -DNDEBUG -O2
# CCOPTSR += -pg
CCOPTSD = ${CCOPTS} -g -O0 -DMKL_DISABLE_FAST_MM=1


srcs  = $(filter-out ${SRC}/main.c, $(wildcard ${SRC}/*.c))
objs  = $(patsubst ${SRC}/%.c, ${OBJ}/%.o,  ${srcs})
zobjs = $(patsubst ${SRC}/%.c, ${OBJ}/z%.o, ${srcs})
objsD = $(patsubst ${SRC}/%.c, ${DBUG}/%.o, ${srcs})
zobjsD= $(patsubst ${SRC}/%.c, ${DBUG}/z%.o,${srcs})

.PHONY: build
build: real complex

.PHONY: real
real: ${BIN}/dmrg tests

.PHONY: complex
complex: ${BIN}/zdmrg ztests

.PHONY: proj_main
proj_main: ${BIN}/dmrg ${BIN}/zdmrg

.PHONY: debug
debug: ${BIN}/dmrg_debug ${BIN}/quick_test_debug ${BIN}/time_test_debug

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

.PHONY: ztests
ztests: $(patsubst $(TEST)/%.c, $(BIN)/z%, $(wildcard $(TEST)/*.c))

.PHONY: quick
quick: bin/quick_test

.PHONY: zquick
zquick: bin/zquick_test

.PHONY: time
time: bin/time_test

.PHONY: ztime
ztime: bin/ztime_test

${OBJ}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSR} ${MKL} $< -o $@

${OBJ}/z%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSR} -DCOMPLEX ${MKL} $< -o $@

${DBUG}/%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSD} ${MKL} $< -o $@

${DBUG}/z%.o: ${SRC}/%.c ${INC}/%.h
	${CC} -c ${INCDIRS} ${CCOPTSD} -DCOMPLEX ${MKL} $< -o $@

${BIN}/dmrg: ${SRC}/main.c ${objs}
	${CC} ${INCDIRS} ${CCOPTSR} ${objs} ${SRC}/main.c ${LIB} -o ${BIN}/dmrg

${BIN}/zdmrg: ${SRC}/main.c ${zobjs}
	${CC} ${INCDIRS} ${CCOPTSR} -DCOMPLEX ${zobjs} ${SRC}/main.c ${LIB} -o ${BIN}/zdmrg

${BIN}/dmrg_debug: ${SRC}/main.c ${objsD}
	${CC} ${INCDIRS} ${CCOPTSD} ${objsD} ${SRC}/main.c ${LIB}  -o ${BIN}/dmrg_debug

${BIN}/quick_test_debug: ${TEST}/quick_test.c ${objsD}
	${CC} ${INCDIRS} ${CCOPTSD} ${objsD} ${TEST}/quick_test.c ${LIB}  -o ${BIN}/quick_test_debug

${BIN}/time_test_debug: ${TEST}/time_test.c ${objsD}
	${CC} ${INCDIRS} ${CCOPTSD} ${objsD} ${TEST}/time_test.c ${LIB}  -o ${BIN}/time_test_debug

${BIN}/zquick_test_debug: ${TEST}/quick_test.c ${zobjsD}
	${CC} ${INCDIRS} ${CCOPTSD} -DCOMPLEX ${zobjsD} ${TEST}/quick_test.c ${LIB} -o ${BIN}/zquick_test_debug
	 
${BIN}/%: ${TEST}/%.c ${objs}
	${CC} ${INCDIRS} ${CCOPTSR} $< ${objs} ${LIB} -o $@

${BIN}/z%: ${TEST}/%.c ${zobjs}
	${CC} ${INCDIRS} ${CCOPTSR} -DCOMPLEX $< ${zobjs} ${LIB} -o $@

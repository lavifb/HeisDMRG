
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
# COMPLEX = true

# compiler options
CCOPTS  = -Wall -xHost -restrict -std=c99 -DMEM_DATA_ALIGN=64

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

build: proj_main test

clean: 
	-rm -rf ${BIN}/*
	-rm ${OBJ}/*

clean-debug:
	-rm -rf ${DBUG}/*

clean-all: clean clean-debug

src : $(filter-out $(OBJ)/main.o,  $(patsubst $(SRC)/%.c, $(OBJ)/%.o,  $(wildcard $(SRC)/*.c)))
srcD: $(filter-out $(DBUG)/main.o, $(patsubst $(SRC)/%.c, $(DBUG)/%.o, $(wildcard $(SRC)/*.c)))

test: $(patsubst $(TEST)/%.c, $(BIN)/%, $(wildcard $(TEST)/[^_]*.c))

${OBJ}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSR} ${LIB} $< -o $@

${DBUG}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSD} ${LIB} $< -o $@

proj_main: src
	 ${CC} ${INCDIRS} ${CCOPTSR} ${LIB} -o ${BIN}/dmrg ${OBJ}/* ${SRC}/main.c

debug: clean-debug srcD
	 ${CC} ${INCDIRS} ${CCOPTSD} ${LIB} -o ${BIN}/dmrg_debug ${DBUG}/* ${SRC}/main.c
	 ${CC} ${INCDIRS} ${CCOPTSD} ${LIB} -o ${BIN}/quick_test_debug ${DBUG}/* ${TEST}/quick_test.c
	 
${BIN}/%: ${TEST}/%.c src
	${CC} ${INCDIRS} ${CCOPTSR} ${LIB} -o $@ $< ${OBJ}/*

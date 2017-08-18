
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

build: proj_main tests

proj_main: ${BIN}/dmrg

clean: 
	-rm -rf ${BIN}/*
	-rm ${OBJ}/*

clean-debug:
	-rm -rf ${DBUG}/*

clean-all: clean clean-debug

src : $(filter-out $(OBJ)/main.c, $(wildcard $(SRC)/*.c))

objs : $(filter-out $(OBJ)/main.o,  $(patsubst $(SRC)/%.c, $(OBJ)/%.o,  $(wildcard $(SRC)/*.c)))
objsD: $(filter-out $(DBUG)/main.o, $(patsubst $(SRC)/%.c, $(DBUG)/%.o, $(wildcard $(SRC)/*.c)))

tests: $(patsubst $(TEST)/%.c, $(BIN)/%, $(wildcard $(TEST)/*.c))

${OBJ}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSR} ${MKL} $< -o $@

${DBUG}/%.o: ${SRC}/%.c
	${CC} -c ${INCDIRS} ${CCOPTSD} ${MKL} $< -o $@

${BIN}/dmrg: objs src
	 ${CC} ${INCDIRS} ${CCOPTSR} -o ${BIN}/dmrg ${OBJ}/* ${SRC}/main.c ${LIB}

debug: clean-debug objsD src
	 ${CC} ${INCDIRS} ${CCOPTSD} -o ${BIN}/dmrg_debug ${DBUG}/* ${SRC}/main.c ${LIB}
	 ${CC} ${INCDIRS} ${CCOPTSD} -o ${BIN}/quick_test_debug ${DBUG}/* ${TEST}/quick_test.c ${LIB}
	 
${BIN}/%: ${TEST}/%.c objs src
	${CC} ${INCDIRS} ${CCOPTSR} -o $@ $< ${OBJ}/* ${LIB}

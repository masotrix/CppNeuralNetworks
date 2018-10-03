CC=g++
CFLAGS=-std=c++14 -g -Isrc/Perceptron
DEPS= src/Perceptron/Perceptron.h
BASEOBJ= src/Perceptron/Perceptron.o
TRAINOBJ= $(BASEOBJ) src/testTrain/testTrain.o
BINARYOPOBJ= $(BASEOBJ) src/testBinaryOp/testBinaryOp.o
SUMGATEOBJ= $(BASEOBJ) src/testSumGate/testSumGate.o
DATAOBJ= src/testTrain/generateData.o

all: build/testTrain/testTrain build/testTrain/generateData \
	build/testBinaryOp/testBinaryOp build/testSumGate/testSumGate

%.o: %.cpp $(DEPS)
	$(CC) $< -c -o $@ $(CFLAGS)

build/testTrain/testTrain: $(TRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

build/testTrain/generateData: $(DATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

build/testBinaryOp/testBinaryOp: $(BINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

build/testSumGate/testSumGate: $(SUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

.PHONY: clean

clean:
	find . -type f -name "*.o" -delete


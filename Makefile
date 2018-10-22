CC=g++
CFLAGS=-std=c++14 -g -Isrc -Isrc/Plot -Isrc/Dataset
CFLAGS+=`pkg-config --cflags --libs opencv`
MKDIR_P= mkdir -p

NEUDEPS= src/Neuron.h src/Plot/Plot.h src/Dataset/Dataset.h
NEUBASEOBJ= src/Neuron.o
PLOTOBJ= src/Plot/Plot.o
DATASETOBJ= src/Dataset/Dataset.o

PERSRC= src/Perceptron
PERBUILD= build/Perceptron
SIGMSRC= src/SigmoidNeuron
SIGMBUILD= build/SigmoidNeuron
NETSRC= src/Network
NETWORKBUILD= build/Network


PERBINARYOPOBJ= $(NEUBASEOBJ) \
								$(PERSRC)/testBinaryOp/testBinaryOp.o
PERSUMGATEOBJ= $(NEUBASEOBJ) \
							 $(SIGMSRC)/testSumGate/testSumGate.o
PERTRAINOBJ= $(NEUBASEOBJ) $(PERSRC)/testTrain/testTrain.o
PERDATAOBJ= $(PERSRC)/testTrain/generateData.o
PERTRAINPLOTOBJ= $(NEUBASEOBJ) $(PLOTOBJ) \
								 $(PERSRC)/testTrain/testTrainPlot.o

SIGMBINARYOPOBJ= $(NEUBASEOBJ) \
								$(SIGMSRC)/testBinaryOp/testBinaryOp.o
SIGMSUMGATEOBJ= $(NEUBASEOBJ) \
								$(SIGMSRC)/testSumGate/testSumGate.o
SIGMTRAINPLOTOBJ= $(NEUBASEOBJ) $(PLOTOBJ) \
								 $(SIGMSRC)/testTrain/testTrainPlot.o
SIGMTRAINOBJ= $(NEUBASEOBJ) $(SIGMSRC)/testTrain/testTrain.o
SIGMDATAOBJ= $(SIGMSRC)/testTrain/generateData.o

NETWORKUNITOBJ = $(NEUBASEOBJ) \
									$(NETSRC)/unitTests.o
NETWORKTRAINOBJ = $(NEUBASEOBJ) $(DATASETOBJ) $(PLOTOBJ) \
									$(NETSRC)/trainNetwork.o
NETWORKTASKONEOBJ = $(NEUBASEOBJ) $(DATASETOBJ) $(PLOTOBJ) \
									$(NETSRC)/task1.o

all: \
	perdirs \
	$(PERBUILD)/testTrain/testTrain \
	$(PERBUILD)/testTrain/generateData \
	$(PERBUILD)/testBinaryOp/testBinaryOp \
	$(PERBUILD)/testSumGate/testSumGate \
	$(PERBUILD)/testTrain/testTrainPlot \
	sigmdirs \
	$(SIGMBUILD)/testTrain/testTrain \
	$(SIGMBUILD)/testTrain/generateData \
	$(SIGMBUILD)/testBinaryOp/testBinaryOp \
	$(SIGMBUILD)/testSumGate/testSumGate \
	$(SIGMBUILD)/testTrain/testTrainPlot \
	netdirs \
	$(NETWORKBUILD)/unitTests/unitTests \
	$(NETWORKBUILD)/testTrain/trainNetwork \
	$(NETWORKBUILD)/task1/task1


%.o: %.cpp $(NEUDEPS)
	$(CC) $< -c -o $@ $(CFLAGS)

builddir:
	${MKDIR_P} build

perdirs: 
	${MKDIR_P} \
  $(PERBUILD)/testBinaryOp \
  $(PERBUILD)/testSumGate \
  $(PERBUILD)/testTrain
$(PERBUILD)/testBinaryOp/testBinaryOp: $(PERBINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(PERBUILD)/testSumGate/testSumGate: $(PERSUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(PERBUILD)/testTrain/testTrain: $(PERTRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(PERBUILD)/testTrain/generateData: $(PERDATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(PERBUILD)/testTrain/testTrainPlot: $(PERTRAINPLOTOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

sigmdirs: 
	${MKDIR_P} \
  $(SIGMBUILD)/testBinaryOp \
  $(SIGMBUILD)/testSumGate \
  $(SIGMBUILD)/testTrain
$(SIGMBUILD)/testBinaryOp/testBinaryOp: $(SIGMBINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(SIGMBUILD)/testSumGate/testSumGate: $(SIGMSUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(SIGMBUILD)/testTrain/testTrain: $(SIGMTRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(SIGMBUILD)/testTrain/generateData: $(SIGMDATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(SIGMBUILD)/testTrain/testTrainPlot: $(SIGMTRAINPLOTOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

netdirs:
	${MKDIR_P} \
  $(NETWORKBUILD)/unitTests \
  $(NETWORKBUILD)/testTrain \
  $(NETWORKBUILD)/task1
$(NETWORKBUILD)/unitTests/unitTests: $(NETWORKUNITOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(NETWORKBUILD)/testTrain/trainNetwork: $(NETWORKTRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)
$(NETWORKBUILD)/task1/task1: $(NETWORKTASKONEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

.PHONY: clean

clean:
	find . -type f -name "*.o" -delete


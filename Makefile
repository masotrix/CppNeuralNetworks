CC=g++
CFLAGS=-std=c++14 -g -INeuron/src -IPlot
CFLAGS+=`pkg-config --cflags --libs opencv`
PERBUILD= Neuron/build/Perceptron
NEUSRC= Neuron/src
PERSRC= Neuron/src/Perceptron
SIGMSRC= Neuron/src/SigmoidNeuron
NEUDEPS= $(NEUSRC)/Neuron.h Plot/Plot.h
NEUBASEOBJ= $(NEUSRC)/Neuron.o
PLOTOBJ= Plot/Plot.o
PERTRAINOBJ= $(NEUBASEOBJ) $(PERSRC)/testTrain/testTrain.o
SIGMTRAINOBJ= $(NEUBASEOBJ) $(SIGMSRC)/testTrain/testTrain.o
PERBINARYOPOBJ= $(NEUBASEOBJ) \
								$(PERSRC)/testBinaryOp/testBinaryOp.o
PERSUMGATEOBJ= $(NEUBASEOBJ) \
							 $(SIGMSRC)/testSumGate/testSumGate.o
SIGMBINARYOPOBJ= $(NEUBASEOBJ) \
								$(SIGMSRC)/testBinaryOp/testBinaryOp.o
SIGMSUMGATEOBJ= $(NEUBASEOBJ) \
								$(SIGMSRC)/testSumGate/testSumGate.o
PERTRAINPLOTOBJ= $(NEUBASEOBJ) $(PLOTOBJ) \
								 $(PERSRC)/testTrain/testTrainPlot.o
SIGMTRAINPLOTOBJ= $(NEUBASEOBJ) $(PLOTOBJ) \
								 $(SIGMSRC)/testTrain/testTrainPlot.o
PERDATAOBJ= $(PERSRC)/testTrain/generateData.o
SIGMDATAOBJ= $(SIGMSRC)/testTrain/generateData.o

all: \
	$(PERBUILD)/testTrain/testTrain \
	$(PERBUILD)/testTrain/generateData \
	$(PERBUILD)/testBinaryOp/testBinaryOp \
	$(PERBUILD)/testSumGate/testSumGate \
	$(PERBUILD)/testTrain/testTrainPlot

%.o: %.cpp $(NEUDEPS)
	$(CC) $< -c -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/testTrain: $(PERTRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/generateData: $(PERDATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testBinaryOp/testBinaryOp: $(PERBINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testSumGate/testSumGate: $(PERSUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/testTrainPlot: $(PERTRAINPLOTOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(SIGMBUILD)/testTrain/testTrain: $(SIGMTRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(SIGMBUILD)/testTrain/generateData: $(SIGMDATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(SIGMBUILD)/testBinaryOp/testBinaryOp: $(SIGMBINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(SIGMBUILD)/testSumGate/testSumGate: $(SIGMSUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(SIGMBUILD)/testTrain/testTrainPlot: $(SIGMTRAINPLOTOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

.PHONY: clean

clean:
	find . -type f -name "*.o" -delete


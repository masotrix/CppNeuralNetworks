CC=g++
CFLAGS=-std=c++14 -g -IPerceptron/src/Perceptron -IPlot
CFLAGS+=`pkg-config --cflags --libs opencv`
PERBUILD= Perceptron/build
PERSRC= Perceptron/src
PERDEPS= $(PERSRC)/Perceptron/Perceptron.h Plot/Plot.h
PERBASEOBJ= $(PERSRC)/Perceptron/Perceptron.o
PLOTOBJ= Plot/Plot.o
TRAINOBJ= $(PERBASEOBJ) $(PERSRC)/testTrain/testTrain.o
BINARYOPOBJ= $(PERBASEOBJ) $(PERSRC)/testBinaryOp/testBinaryOp.o
SUMGATEOBJ= $(PERBASEOBJ) $(PERSRC)/testSumGate/testSumGate.o
PERTRAINPLOTOBJ= $(PERBASEOBJ) $(PLOTOBJ) \
								 $(PERSRC)/testTrain/testTrainPlot.o
DATAOBJ= $(PERSRC)/testTrain/generateData.o

all: \
	$(PERBUILD)/testTrain/testTrain \
	$(PERBUILD)/testTrain/generateData \
	$(PERBUILD)/testBinaryOp/testBinaryOp \
	$(PERBUILD)/testSumGate/testSumGate \
	$(PERBUILD)/testTrain/testTrainPlot

%.o: %.cpp $(PERDEPS)
	$(CC) $< -c -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/testTrain: $(TRAINOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/generateData: $(DATAOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testBinaryOp/testBinaryOp: $(BINARYOPOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testSumGate/testSumGate: $(SUMGATEOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

$(PERBUILD)/testTrain/testTrainPlot: $(PERTRAINPLOTOBJ)
	$(CC) $^ -o $@ $(CFLAGS)

.PHONY: clean

clean:
	find . -type f -name "*.o" -delete


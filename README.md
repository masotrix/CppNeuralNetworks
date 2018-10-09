# CppNeuralNetworks

<b>Autor:</b> Ignacio Matías Haeussler Risco

<b>Perceptrón y Sigmoid Neuron</b>

1. Implementación de Perceptrón y Sigmoid neuron, clases hijas y gates:
  - /Neuron/src/Neuron.h
  - /Neuron/src/Neuron.cpp
  
2. Implementación funcionalidad para graficar:
  - /Plot/Plot.h
  - /Plot/Plot.cpp

2. Compilación:
  - /Makefile

3. Ejecutables perceptrón:
  - /Neuron/build/Perceptron

4. Ejecutables sigmoid neuron:
  - /Neuron/build/SigmoidNeuron

## Ejercicio 1 (Perceptron binary operations)

Implementación de Testeo de operaciones binarias (AND,OR,NAND,SUM): 
  - /Neuron/src/Perceptron/testBinaryOp/testBinaryOp.cpp

Implementación de Testeo de sum gate (XOR): 
  - /Neuron/src/Perceptron/testSumGate/testSumGate.cpp

Ejecutable para testear operaciones bianarias:
  - /Neuron/build/Perceptron/testBinaryOp/testBinaryOp

Ejecutable para testear sum gate:
  - /Neuron/build/Perceptron/testSumGate/testSumGate
  
Conclusiones:
  - El perceptron es capaz de implementar compuertas lógicas
  simples como AND, OR y NAND. Por otro lado, más de uno deben
  ser combinados para generar compuertas más complejas como XOR.

## Ejercicio 2 (Perceptron Training)

Implementación gráficos de distribución y entrenamiento
  - /Neuron/src/Perceptron/testTrain/testTrainPlot.cpp
  
Ejecutable para generar gráficos de distribución y entrenamiento
  - /Neuron/build/Perceptron/testTrain/testTrainPlot
  
Conclusions:
  - El perceptrón no parece ser afectado por cambios en la tasa de
  aprendizaje.
  
## Ejercicio 3 (Sigmoid Neuron binary operations and training)

Implementación de Testeo de operaciones binarias (AND,OR,NAND,SUM): 
  - /Neuron/src/SigmoidNeuron/testBinaryOp/testBinaryOp.cpp

Implementación de Testeo de sum gate (XOR): 
  - /Neuron/src/SigmoidNeuron/testSumGate/testSumGate.cpp

Ejecutable para testear operaciones bianarias:
  - /Neuron/build/SigmoidNeuron/testBinaryOp/testBinaryOp

Ejecutable para testear sum gate:
  - /Neuron/build/SigmoidNeuron/testSumGate/testSumGate
  
Implementación gráficos de distribución y entrenamiento
  - /Neuron/src/SigmoidNeuron/testTrain/testTrainPlot.cpp
  
Ejecutable para generar gráficos de distribución y entrenamiento
  - /Neuron/build/SigmoidNeuron/testTrain/testTrainPlot

Conclusions:
  - La neurona sigmoide es capaz de implementar compuertas lógicas
  simples y, al combinar más de una, compuertas lógicas complejas.
  - El aprendizaje de la neurona sigmoide demuestra una sensibilidad
  muy alta a la tasa de aprendizaje, contrariamente al perceptrón.

# CppNeuralNetworks

<b>Autor:</b> Ignacio Matías Haeussler Risco

<b>Perceptrón y Sigmoid Neuron</b>

1. Implementación de Perceptrón y Sigmoid neuron, clases hijas y gates:
  - /src/Neuron/src/Neuron.h
  - /src/Neuron/src/Neuron.cpp
  
2. Implementación funcionalidad para graficar:
  - /src/Plot/Plot.h
  - /src/Plot/Plot.cpp

3. Compilación (generan carpeta /build):
  - /Makefile

4. Ejecutables perceptrón:
  - /build/Perceptron

5. Ejecutables sigmoid neuron:
  - /build/SigmoidNeuron
  
6. Ejecutables red neuronal:
  - /build/Network
  
<b>Notas</b>

  - C++ OpenCV 3.4 es utilizado para realizar gráficos.
  - Ejecutables de testeo entregan output si hubo algún error.
  - Ejecutables solo disponibles luego de compilación
  - Makefile asume ambiente Unix

## Ejercicio 1 (Perceptron binary operations)

Implementación de Testeo de operaciones binarias (AND,OR,NAND,SUM): 
  - /src/Perceptron/testBinaryOp/testBinaryOp.cpp

Implementación de Testeo de sum gate (XOR): 
  - /src/Perceptron/testSumGate/testSumGate.cpp

Ejecutable para testear operaciones bianarias:
  - /build/Perceptron/testBinaryOp/testBinaryOp

Ejecutable para testear sum gate:
  - /build/Perceptron/testSumGate/testSumGate
  
Conclusiones:
  - El perceptron es capaz de implementar compuertas lógicas
  simples como AND, OR y NAND. Por otro lado, más de uno deben
  ser combinados para generar compuertas más complejas como XOR.

## Ejercicio 2 (Perceptron Training)

Implementación gráficos de distribución y entrenamiento
  - /src/Perceptron/testTrain/testTrainPlot.cpp
  
Ejecutable para generar gráficos de distribución y entrenamiento
  - /build/Perceptron/testTrain/testTrainPlot
  
Conclusiones:
  - El perceptrón no parece ser afectado por cambios en la tasa de
  aprendizaje.
  
## Ejercicio 3 (Sigmoid Neuron binary operations and training)

Implementación de Testeo de operaciones binarias (AND,OR,NAND,SUM): 
  - /src/SigmoidNeuron/testBinaryOp/testBinaryOp.cpp

Implementación de Testeo de sum gate (XOR): 
  - /src/SigmoidNeuron/testSumGate/testSumGate.cpp

Ejecutable para testear operaciones bianarias:
  - /build/SigmoidNeuron/testBinaryOp/testBinaryOp

Ejecutable para testear sum gate:
  - /build/SigmoidNeuron/testSumGate/testSumGate
  
Implementación gráficos de distribución y entrenamiento
  - /src/SigmoidNeuron/testTrain/testTrainPlot.cpp
  
Ejecutable para generar gráficos de distribución y entrenamiento
  - /build/SigmoidNeuron/testTrain/testTrainPlot

Conclusiones:
  - La neurona sigmoide es capaz de implementar compuertas lógicas
  simples y, al combinar más de una, compuertas lógicas complejas.
  - El aprendizaje de la neurona sigmoide demuestra una sensibilidad
  muy alta a la tasa de aprendizaje, contrariamente al perceptrón.
  
## Ejercicio 4 (Neural Network)

Implementación gráfico de entrenamiento (loss)
  - /src/Network/trainNetwork.cpp
  
Ejecutable gráfico de entrenamiento
  - /build/Network/testTrain/trainNetwork
  
  

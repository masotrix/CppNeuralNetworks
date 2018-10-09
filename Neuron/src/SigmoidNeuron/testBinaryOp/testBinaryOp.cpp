#include <vector>
#include <iostream>
#include <Neuron.h>
#include <functional>
#include <memory>
#include <cmath>
using namespace std;

int main() {

  vector<vector<float>> bitwiseX =
    {{0,0},{0,1},{1,0},{1,1}};
  
  auto andNeu = make_unique<ANDSigmoidNeuron>();
  andNeu->test(bitwiseX, {0.0,0.0,0.0,1.0}, true, "&& Test");

  auto orNeu = make_unique<ORSigmoidNeuron>();
  orNeu->test(bitwiseX, {0.0,1.0,1.0,1.0}, true, "|| Test");

  auto nandNeu = make_unique<NANDSigmoidNeuron>();
  nandNeu->test(bitwiseX, {1.0,1.0,1.0,0.0}, true, "NAND Test");
}

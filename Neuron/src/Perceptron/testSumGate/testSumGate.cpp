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

  auto xorGate = make_unique<PerceptronXORGate>();
  xorGate->test(bitwiseX, {0.0,1.0,1.0,0.0}, true);
}

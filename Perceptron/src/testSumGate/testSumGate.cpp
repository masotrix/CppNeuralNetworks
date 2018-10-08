#include <vector>
#include <iostream>
#include <Perceptron.h>
#include <functional>
#include <memory>
#include <cmath>
using namespace std;

int main() {

  vector<vector<float>> bitwiseX =
    {{0,0},{0,1},{1,0},{1,1}};

  unique_ptr<XORGate> xorGate(new XORGate());
  xorGate->test(bitwiseX, {0.0,1.0,1.0,0.0}, true);
}

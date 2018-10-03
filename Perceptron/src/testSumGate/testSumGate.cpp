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


  cout << "Begin Bitwise Tests:\n\n";

  cout << "&& Tests: " << endl;
  unique_ptr<XORGate> xorGate(new XORGate());
  float xorR = 100* xorGate->test(bitwiseX, {0.0,1.0,1.0,0.0});
  cout << "&& Coverage: " << xorR << "%\n\n";
}

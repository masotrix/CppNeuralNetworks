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
  
  auto andPer = make_unique<ANDPerceptron>();
  andPer->test(bitwiseX,{0,0,0,1},true,"&& Test");

  auto orPer = make_unique<ORPerceptron>();
  orPer->test(bitwiseX,{0,1,1,1}, true, "|| Test");

  auto nandPer = make_unique<NANDPerceptron>();
  nandPer->test(bitwiseX,{1,1,1,0}, true, "NAND Test");
}

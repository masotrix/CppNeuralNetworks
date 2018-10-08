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
  
  unique_ptr<ANDPerceptron> andPer(new ANDPerceptron());
  andPer->test(bitwiseX, {0.0,0.0,0.0,1.0}, true, "&& Test");

  unique_ptr<ORPerceptron> orPer(new ORPerceptron());
  orPer->test(bitwiseX, {0.0,1.0,1.0,1.0}, true, "|| Test");

  unique_ptr<NANDPerceptron> nandPer(new NANDPerceptron());
  nandPer->test(bitwiseX, {1.0,1.0,1.0,0.0}, true, "NAND Test");
}

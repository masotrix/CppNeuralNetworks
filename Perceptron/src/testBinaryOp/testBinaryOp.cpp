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

  vector<vector<float>> sumX =
    {{1.0,-1.0},{-1.0,1.0},{1.0,2.0},{2.0,1.0},
    {12.345,-87.654}};
  

  cout << "Begin Bitwise Tests:\n\n";

  cout << "&& Tests: " << endl;
  unique_ptr<ANDPerceptron> andPer(new ANDPerceptron());
  float andR = 100*andPer->test(bitwiseX, {0.0,0.0,0.0,1.0});
  cout << "&& Coverage: " << andR << "%\n\n";

  cout << "|| Tests: " << endl;
  unique_ptr<ORPerceptron> orPer(new ORPerceptron());
  float orR = 100*orPer->test(bitwiseX, {0.0,1.0,1.0,1.0});
  cout << "|| Coverage: " << orR << "%\n\n";

  cout << "NAND Tests: " << endl;
  unique_ptr<NANDPerceptron> nandPer(new NANDPerceptron());
  float nandR = 100*nandPer->test(bitwiseX, {1.0,1.0,1.0,0.0});
  cout << "NAND Coverage: " << nandR << "%\n\n";

  cout << "Begin Standard Arithmetic Tests:\n\n";

  cout << "+ Tests: " << endl;
  unique_ptr<SUMPerceptron> sumPer(new SUMPerceptron());
  float sumR = 100*sumPer->test(sumX, {0,0,3,3,-75.309});
  cout << "SUM Coverage: " << sumR << "%\n\n";
}

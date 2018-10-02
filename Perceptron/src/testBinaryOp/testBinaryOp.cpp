#include <vector>
#include <iostream>
#include <Perceptron.h>
#include <functional>
#include <memory>
#include <cmath>
using namespace std;

int main() {

  function<float(float,float)> op;
  function<float(float)> floorAct =
        [](float s)->float{ return floor(s); };
  function<float(float)> idenAct =
        [](float s)->float{ return s; };

  auto test = Perceptron::testOp;

  vector<vector<float>> bitwiseX =
    {{0,0},{0,1},{1,0},{1,1}};

  vector<vector<float>> sumX =
    {{1.0,-1.0},{-1.0,1.0},{1.0,2.0},{2.0,1.0},
    {12.345,-87.654}};
  

  cout << "Begin Bitwise Tests:\n\n";

  cout << "&& Tests: " << endl;
  op = [](float a, float b) {return (float)(a&&b);};
  float andR = 100*test({0.5,0.5},0.0,floorAct,op,bitwiseX);
  cout << "&& Coverage: " << andR << "%\n\n";

  cout << "|| Tests: " << endl;
  op = [](float a, float b) {return (float)(a||b);};
  float orR = 100*test({0.5,0.5},0.5,floorAct,op,bitwiseX);
  cout << "|| Coverage: " << orR << "%\n\n";

  cout << "NAND Tests: " << endl;
  op = [](float a, float b) {return (float)(!(a&&b));};
  float nandR = 100*test({-0.5,-0.5},1.5,floorAct,op,bitwiseX);
  cout << "NAND Coverage: " << nandR << "%\n\n";

  cout << "+ Tests: " << endl;
  op = [](float a, float b) {return (float)(a+b);};
  float sumR = 100*test({1.0,1.0},0.0,idenAct,op,sumX);
  cout << "SUM Coverage: " << sumR << "%\n\n";
}

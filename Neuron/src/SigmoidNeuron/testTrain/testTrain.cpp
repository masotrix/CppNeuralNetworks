#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include <Perceptron.h>
using namespace std;

#define trainN 1000


int main() {

  vector<vector<vector<float>>> X(2,vector<vector<float>>());
  vector<vector<float>> Y(2,vector<float>());
  vector<string> fnames(2);
  fnames[0] = "trainData.txt"; fnames[1]="testData.txt";
  for (int k=0; k<2; k++) {
    ifstream ifile(fnames[k]);
    for (int i=0; i<trainN; i++) {
      vector<float> point(2);
      ifile >> point[0]; ifile >> point[1];
      if (point[1] < 0) Y[k].push_back(-1);
      else Y[k].push_back(1);
      X[k].push_back(point);
    }
    ifile.close();
  }


  unique_ptr<TwoInputPerceptron> per(new TwoInputPerceptron());
  vector<float> loss; int EPOCHS = 20;
  per->train(X[0],Y[0],0.00001,EPOCHS,loss);
  
  ofstream ofile("loss.txt");
  for (int i=0; i<loss.size(); i++) {
    ofile << i+1;
    ofile << ' ';
    ofile << loss[i];
    ofile << endl;
  }
  ofile.close();

  cout << per->test(X[1], Y[1]) << endl;
}

#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include <Neuron.h>
using namespace std;

#define trainN 1000


int main() {

  vector<vector<pair<vector<float>,float>>> X(2);
  vector<string> fnames(2);
  fnames[0] = "trainData.txt"; fnames[1]="testData.txt";
  for (int k=0; k<2; k++) {
    ifstream ifile(fnames[k]);
    for (int i=0; i<trainN; i++) {
      float y;
      vector<float> point(2);
      ifile >> point[0]; ifile >> point[1];
      if (point[1] < 0) y = -1;
      else y= 1;
      X[k].push_back(make_pair(point,y));
    }
    ifile.close();
  }

  vector<vector<pair<vector<float>,float>>> Xtr;
  for (int i=0; i<X[0].size(); i++)
    Xtr.push_back({X[0][i]});


  unique_ptr<TwoInputPerceptron> per(new TwoInputPerceptron());
  vector<float> loss; int EPOCHS = 20;
  per->train(Xtr,0.00001,EPOCHS,loss);
  
  ofstream ofile("loss.txt");
  for (int i=0; i<loss.size(); i++) {
    ofile << i+1;
    ofile << ' ';
    ofile << loss[i];
    ofile << endl;
  }
  ofile.close();

  cout << per->test(X[1]) << endl;
}

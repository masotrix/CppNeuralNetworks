#include <Dataset.h>
#include <random>
#include <functional>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
using namespace std;

vector<vector<pair<vector<float>,vector<float>>>>
  Dataset::getTrainingExamples(int bsize, bool shuff) {

  default_random_engine gen;
  if (shuff) shuffle(_XTR.begin(),_XTR.end(),gen);
  vector<vector<pair<vector<float>,vector<float>>>> X(
      _XTR.size()/bsize+(_XTR.size()%bsize?1:0));
  int belems=0, b=0;
  for (int i=0; i<_XTR.size(); i++) {
    X[b].push_back(_XTR[i]);
    belems++;
    if (belems==bsize) {
      belems=0; b++;
    }
  }


  return X; 
}

const vector<pair<vector<float>,vector<float>>>
  Dataset::getTestingExamples() { return _XTR; }

RandomDataset::RandomDataset(const int N, const int F,
    const int C) {

  default_random_engine gen;
  normal_distribution<float> ndist(0.f, 1.f);
  uniform_int_distribution<int> udist(0,C-1);

  for (int i=0; i<N; i++) {
    std::vector<float> x,y(C);
    int clss = udist(gen);
    for (int j=0; j<F; j++)
      x.push_back(ndist(gen));
    for (int j=0; j<C; j++)
      clss==j? y[j]=1.f:y[j]=0.f;
    _XTR.push_back(make_pair(x,y));
  }    
}

void UnseparatedDataset::separateData(
    const vector<pair<vector<float>,vector<float>>> &X,
    float testProportion) {

  default_random_engine gen;
  uniform_real_distribution<> dist(0.0,1.0);
  auto coin = bind(dist,gen);

  for (int i=0; i<X.size(); i++) {
    if (coin()>testProportion)
      _XTR.push_back(X[i]);
    else _XTE.push_back(X[i]);
  }
}

const vector<pair<vector<float>,vector<float>>>
  UnseparatedDataset::getTestingExamples() { return _XTE; }

UnseparatedCSVDataset::UnseparatedCSVDataset(string csvfileName,
    float testProportion) {
  
  string s, token; istringstream iss;
  vector<pair<vector<float>,vector<float>>> X;
  ifstream csvfile(csvfileName);
  
  while (getline(csvfile, s)) {
    vector<float> x(10,0),y,lineEls;
    iss.clear(); iss.str(s);
    getline(iss, token, ',');
    switch (token[0]) {
      case 'L': y={1,0,0}; break;
      case 'B': y={0,1,0}; break;
      case 'R': y={0,0,1}; break;
    }
    while (getline(iss, token, ','))
      lineEls.push_back(stof(token));
    x[lineEls[1]-1]=lineEls[0];
    x[lineEls[3]+4]=lineEls[2];
    X.push_back(make_pair(x,y));
  }

  csvfile.close();
  separateData(X,testProportion);
}


#include <Dataset.h>
using namespace std;

pair<vector<vector<float>>,vector<vector<float>>>
  Dataset::getTrainingExamples() {
  return make_pair(_XTR,_YTR);
}
pair<vector<vector<float>>,vector<vector<float>>> 
  Dataset::getTestingExamples() {
  return make_pair(_XTE,_YTE);
}

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
    _XTR.push_back(x);
    _YTR.push_back(y);
  }    
}

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <utility>
using namespace std;

#define trainN 1000

int main() {
  
  float moonR=10, moonW=6, moonIR=moonR-moonW, d=1;
  pair<float,float> moonB; moonB.first=moonR; moonB.second=-d;

  srand(time(NULL));


  vector<vector<pair<float,float>>> points(2,
      vector<pair<float,float>>());
  float r, th;
  vector<string> fnames;
  fnames.push_back("trainData.txt");
  fnames.push_back("testData.txt");

  for (int k=0; k<2; k++) {
    for (int i=0; i<trainN; i++) {
      pair<float,float> point;
      r = moonIR + (float)rand()/(float)RAND_MAX*moonW;
      th = 0;
      if (rand() > RAND_MAX/2) {
        th = M_PI;
        point.first = moonB.first;
        point.second = moonB.second;
      }
      th += (float)rand()/(float)RAND_MAX*M_PI;
      point.first += r*cos(th);
      point.second += r*sin(th);

      points[k].push_back(point);
    }
    ofstream file(fnames[k]);
    for (int i=0; i<points[k].size(); i++) {
      file << points[k][i].first;
      file << ' ';
      file << points[k][i].second;
      file << endl;
    }
    file.close();
  }
}

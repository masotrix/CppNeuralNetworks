#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <cmath>
#include <Neuron.h>
#include <Plot.h>
#include <iomanip>
#include <sstream>
using namespace std;
using namespace cv;

#define TRAIN_SAMPLES 500
#define TEST_SAMPLES 50
#define EPOCHS 20

int main() {

  vector<vector<vector<float>>> X(2);
  vector<vector<float>> Y(2,vector<float>());
  vector<int> samples = {TRAIN_SAMPLES, TEST_SAMPLES};
  float w1=-1.0,w2=2.0,b=-5;

  srand(time(NULL));
  ofstream out("data.txt");
  for (int k=0; k<2; k++) {
    for (int i=0; i<samples[k]; i++) {
      vector<float> x = vector<float>{
        100.f*(rand()/(float)RAND_MAX-0.5f),
        100.f*(rand()/(float)RAND_MAX-0.5f)};
      if (w1*x[0]+w2*x[1]+b>0)
        Y[k].push_back(1);
      else Y[k].push_back(0);
      X[k].push_back(x);
    }
  }

  vector<vector<vector<float>>> zeros(
      4,vector<vector<float>>()),ones(
        4,vector<vector<float>>());

  auto aper = make_unique<TwoInputPerceptron>();
  vector<float> aloss; float alr = 0.0001;
  vector<float> classTests = {2, 5, 10, 50}; 
  int iniIte = 0;
  for (int i=0; i<classTests.size(); i++) {
    for (int j=iniIte; j<classTests[i]; j++) {
      aper->train({X[0][j]},{Y[0][j]},alr,1,aloss);
    }

    for (int j=0; j<samples[0]; j++) {
      if (aper->forward(X[0][j]))
        ones[i].push_back(X[0][j]);
      else zeros[i].push_back(X[0][j]);
    }
    
    iniIte = classTests[i];
  }
  

  stringstream ss; ss.str(string());
  ss << scientific << setprecision(1.f);

  unique_ptr<Plotter> dataDistPlotter(new Plotter());
  for (int i=0; i<4; i++) {
    shared_ptr<Plot> dataDistChart(new Plot());
    dataDistChart->addPlotData(zeros[i],Scalar(255,0,0), false);
    dataDistChart->addPlotData(ones[i],Scalar(0,0,255), false);
    float m=-w1/w2, n=-b, leftX=-50,rightX=50,
      leftY = m*leftX+n, rightY=m*rightX+n;
    dataDistChart->addPlotData({{leftX,leftY},{rightX,rightY}});
    dataDistChart->setTitle(string("Data distribution ")+
        string("for ")+to_string((int)classTests[i])+
        string(" training iterations"));
    dataDistChart->setLimits({-50,50,-120,120});
    dataDistPlotter->addPlot(dataDistChart);
  }
  ss << alr;
  dataDistPlotter->plot(string("Data distributions for ")+
      string("learning rate ")+ss.str());
  ss.str(string());


  vector<unique_ptr<TwoInputPerceptron>> pers;
  vector<vector<float>> loss(4,vector<float>()),
      accs(4,vector<float>());
  vector<float> lrs = {0.0001,0.001,0.01,1.0};
  for (int i=0; i<4; i++) {
    pers.push_back(make_unique<TwoInputPerceptron>());
  }
  for (int i=0; i<samples[1]; i++) {
    for (int j=0; j<4; j++) {
      pers[j]->train({X[0][i]},{Y[0][i]},lrs[j],1,loss[j]);
      accs[j].push_back(pers[j]->test(X[0],Y[0]));
    }
  }

  vector<vector<vector<float>>> 
    lossCoords(4, vector<vector<float>>()),
    accCoords(4, vector<vector<float>>());
  for (int i=0; i<4; i++)
    for (int j=0; j<accs[i].size(); j++)
      accCoords[i].push_back({(float)j,accs[i][j]});

  unique_ptr<Plotter> accPlotter(new Plotter());
  for (int i=0; i<4; i++) {
    shared_ptr<Plot> accChart(new Plot());
    accChart->addPlotData(accCoords[i],Scalar(255,0,0));
    accChart->setTicks(5,4);
    accChart->setXLabel("Train Iteration");
    accChart->setYLabel("Accuracy");
    ss << lrs[i];
    accChart->setTitle(string("Accuracy vs Train Iteration, ") +
        string("lr=") + ss.str());
    ss.str(string());
    accChart->setLimits({0,(float)samples[1],0,1});
    accPlotter->addPlot(accChart);
  }
  accPlotter->plot(string("Accuracy vs Iteration for different ")+
      string("learning rates"));

  //for (int i=0; i<loss.size(); i++)
    //lossCoords.push_back({(float)i,loss[i]});
  /*
  unique_ptr<Plot> lossChart(new Plot());
  //lossChart->addPlotData(lossCoords,Scalar(255,0,0));
  lossChart->addPlotData(accCoords,Scalar(255,0,0));
  lossChart->setTicks(5,4);
  lossChart->setXLabel("Train Iteration");
  lossChart->setYLabel("Accuracy");
  lossChart->setTitle("Accuracy vs Train Iteration");
  lossChart->plot("OpenCV Plot",{0,(float)samples[1],0,1});
  */
}

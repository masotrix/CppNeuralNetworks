#include <memory>
#include <Dataset.h>
#include <Neuron.h>
#include <Plot.h>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

static void printVec(const vector<float> &x) {
  for (float f: x) cout << f << ' ';
  cout << endl;
}

int main() {

  int F=300,C=10,h1=50,h2=30,N=100,EPOCHS=30,B=1;
  //int F=2,C=2,h1=2,h2=2,N=1,EPOCHS=1,B=1;
  auto rd = make_unique<RandomDataset>(N,F,C);
  auto sig1 = make_unique<SigmoidLayer>(F,h1);
  auto sig2 = make_unique<SigmoidLayer>(h1,h2);
  //auto sig1 = make_unique<ReluLayer>(F,h1);
  //auto sig2 = make_unique<ReluLayer>(h1,h2);
  auto soft = make_unique<SoftMaxLayer>(h2,C);
  vector<unique_ptr<NeuralLayerInterface>> layers;
  layers.push_back(move(sig1));
  layers.push_back(move(sig2));
  layers.push_back(move(soft));
  auto nn = make_unique<NeuralNetwork>(move(layers));

  vector<vector<vector<float>>> weights;
  vector<vector<pair<vector<float>,vector<float>>>> X;
  X = rd->getTrainingExamples(2);
  /*weights = {{{-0.4075,0.7333},{-1.0264,0.7863}},
   {{-0.2433,1.2113},{0.1963,-0.8431}},
   {{-0.7903,-0.1687},{-1.1256,-0.0185}}};
  X = {{-0.4858635663986206, 0.007866157218813896},
    {-0.8678123354911804, -2.4042041301727295},
    {1.5137875080108643, 0.8375995755195618}};
  Y = {{1,0},{0,1},{1,0}};*/

  vector<float> loss;
  nn->train(X,0.1,EPOCHS,loss);

  //cout << nn->test(X,Y) << endl;
  //vector<float> y_pred = nn->forward({X[0]});

  vector<vector<float>> lossCoords;
  for (int j=0; j<loss.size(); j++)
    lossCoords.push_back({(float)j,loss[j]});
  //cout << lossCoords.size() << endl;

  unique_ptr<Plotter> lossPlotter(new Plotter());
    shared_ptr<Plot> lossChart(new Plot());
    lossChart->addPlotData(lossCoords,Scalar(255,0,0));
    lossChart->setTicks(5,4);
    lossChart->setXLabel("Train Iteration");
    lossChart->setYLabel("Loss");
    lossChart->setTitle(string("Loss vs Train Iteration"));
    lossChart->setLimits({0,(float)loss.size(),0,5.0});
    lossPlotter->addPlot(lossChart);
  lossPlotter->plot(
      string("Loss vs Iteration for different"));
}

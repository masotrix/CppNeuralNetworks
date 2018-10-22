#include <memory>
#include <Dataset.h>
#include <Neuron.h>
#include <Plot.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

static void printVec(const vector<float> &x) {
  for (float f: x) cout << f << ' ';
  cout << endl;
}

int main() {

  float prop = 0.2;
  chrono::time_point<chrono::system_clock> start,end;
  chrono::duration<double> seconds;

  { // Show different layers

    int F=10,C=3,h1=5,h2=3,EPOCHS=100,B=10;
    float lr=0.25, acc1, acc2, acc3, acc4,
          maxi1=0, maxi2=0, maxi3=0, maxi4=0, maxi;
    vector<float> loss1,loss2,loss3,loss4;
    vector<vector<float>>
      lossCoords1, lossCoords2, lossCoords3, lossCoords4;
    vector<unique_ptr<NeuralLayerInterface>>
      layers1, layers2, layers3, layers4;
    auto ds = make_unique<UnseparatedCSVDataset>(
        "data/task1.txt", prop);
    vector<vector<pair<vector<float>,vector<float>>>> Xtr;
    vector<pair<vector<float>,vector<float>>> Xte;
    Xtr = ds->getTrainingExamples(B);
    Xte = ds->getTestingExamples();

    auto soft1 = make_unique<SoftMaxLayer>(F,C);
    layers1.push_back(move(soft1));
    auto nn1 = make_unique<NeuralNetwork>(move(layers1));
    start = chrono::system_clock::now();
    nn1->train(Xtr,lr,EPOCHS,loss1,true);
    end = chrono::system_clock::now();
    seconds = end-start;
    cout << "Time conf1: " << seconds.count() << " seconds\n";
    for (int i=0; i<loss1.size(); i++) maxi1=max(maxi1,loss1[i]);
    acc1 = 100*nn1->test(Xte);
    for (int j=0; j<loss1.size(); j++)
      lossCoords1.push_back({(float)j,loss1[j]});

    auto sig2 = make_unique<SigmoidLayer>(F,h2);
    auto soft2 = make_unique<SoftMaxLayer>(h2,C);
    layers2.push_back(move(sig2));
    layers2.push_back(move(soft2));
    auto nn2 = make_unique<NeuralNetwork>(move(layers2));
    start = chrono::system_clock::now();
    nn2->train(Xtr,lr,EPOCHS,loss2,true);
    end = chrono::system_clock::now();
    seconds = end-start;
    cout << "Time conf2: " << seconds.count() << " seconds\n";
    for (int i=0; i<loss2.size(); i++) maxi2=max(maxi2,loss2[i]);
    acc2 = 100*nn2->test(Xte);
    for (int j=0; j<loss2.size(); j++)
      lossCoords2.push_back({(float)j,loss2[j]});

    auto sig31 = make_unique<SigmoidLayer>(F,h1);
    auto sig32 = make_unique<SigmoidLayer>(h1,h2);
    auto soft3 = make_unique<SoftMaxLayer>(h2,C);
    layers3.push_back(move(sig31));
    layers3.push_back(move(sig32));
    layers3.push_back(move(soft3));
    auto nn3 = make_unique<NeuralNetwork>(move(layers3));
    start = chrono::system_clock::now();
    nn3->train(Xtr,lr,EPOCHS,loss3,true);
    end = chrono::system_clock::now();
    seconds = end-start;
    cout << "Time conf3: " << seconds.count() << " seconds\n";
    for (int i=0; i<loss3.size(); i++) maxi3=max(maxi3,loss3[i]);
    acc3 = 100*nn3->test(Xte);
    for (int j=0; j<loss3.size(); j++)
      lossCoords3.push_back({(float)j,loss3[j]});

    auto sig41 = make_unique<SigmoidLayer>(F,h1);
    auto sig42 = make_unique<SigmoidLayer>(h1,h1);
    auto sig43 = make_unique<SigmoidLayer>(h1,h2);
    auto soft4 = make_unique<SoftMaxLayer>(h2,C);
    layers4.push_back(move(sig41));
    layers4.push_back(move(sig42));
    layers4.push_back(move(sig43));
    layers4.push_back(move(soft4));
    auto nn4 = make_unique<NeuralNetwork>(move(layers4));
    start = chrono::system_clock::now();
    nn4->train(Xtr,lr,EPOCHS,loss4,true);
    end = chrono::system_clock::now();
    seconds = end-start;
    cout << "Time conf4: " << seconds.count() << " seconds\n";
    for (int i=0; i<loss4.size(); i++) maxi4=max(maxi4,loss4[i]);
    acc4 = 100*nn4->test(Xte);
    for (int j=0; j<loss4.size(); j++)
      lossCoords4.push_back({(float)j,loss4[j]});

    stringstream ss; ss.str(string());
    ss << fixed << setprecision(1.f);
    maxi = max(maxi1, maxi2); maxi = max(maxi, maxi3);
    maxi = max(maxi, maxi4);
    unique_ptr<Plotter> lossPlotter(new Plotter());
      shared_ptr<Plot> lossChart1(new Plot());
      lossChart1->addPlotData(lossCoords1,Scalar(255,0,0));
      lossChart1->setTicks(5,4);
      lossChart1->setXLabel("Train Iteration");
      lossChart1->setYLabel("Loss");
      ss << acc1;
      lossChart1->setTitle(string("NN[10,soft:3] ")+
          string("(Test Acc: ")+
          ss.str()+string("%)")); ss.str(string());
      lossChart1->setLimits({0,(float)loss1.size(),0,maxi});
      lossPlotter->addPlot(lossChart1);

      shared_ptr<Plot> lossChart2(new Plot());
      lossChart2->addPlotData(lossCoords2,Scalar(255,0,0));
      lossChart2->setTicks(5,4);
      lossChart2->setXLabel("Train Iteration");
      lossChart2->setYLabel("Loss");
      ss << acc2;
      lossChart2->setTitle(string("NN[10,sig:3,soft:3] ")+
         string("(Test Acc: ")+
          ss.str()+string("%)")); ss.str(string());
      lossChart2->setLimits({0,(float)loss2.size(),0,maxi});
      lossPlotter->addPlot(lossChart2);

      shared_ptr<Plot> lossChart3(new Plot());
      lossChart3->addPlotData(lossCoords3,Scalar(255,0,0));
      lossChart3->setTicks(5,4);
      lossChart3->setXLabel("Train Iteration");
      lossChart3->setYLabel("Loss");
      ss << acc3;
      lossChart3->setTitle(string("NN[10,sig:5,sig:3,soft:3] ")+
         string("(Test Acc: ")+
          ss.str()+string("%)")); ss.str(string());
      lossChart3->setLimits({0,(float)loss3.size(),0,maxi});
      lossPlotter->addPlot(lossChart3);

      shared_ptr<Plot> lossChart4(new Plot());
      lossChart4->addPlotData(lossCoords4,Scalar(255,0,0));
      lossChart4->setTicks(5,4);
      lossChart4->setXLabel("Train Iteration");
      lossChart4->setYLabel("Loss");
      ss << acc4;
      lossChart4->setTitle(
          string("NN[10,2xsig:5.sig:3,soft:3] ")+
          string("(Test Acc: ")+
          ss.str()+string("%)")); ss.str(string());
      lossChart4->setLimits({0,(float)loss4.size(),0,maxi});
      lossPlotter->addPlot(lossChart4);


    lossPlotter->plot(
        string("Loss vs Iteration for different neural layer ")+
        string("configuartions (algorithm: SGD, ")+
        string("learning rate: 0.25, Batch Size:10, ")+
        string("EPOCHS: 100)"));
  }

  { // Show different learning rates

    int F=10,C=3,h=3,EPOCHS=100,B=10;
    float lr1=0.001, lr2=0.01, lr3=0.1, lr4=1.0, maxi,
          acc1, acc2, acc3, acc4, maxi1, maxi2, maxi3, maxi4;
    maxi1=maxi2=maxi3=maxi4=0.0;
    vector<float> loss1,loss2,loss3,loss4;
    vector<vector<float>> lossCoords1, lossCoords2,lossCoords3,
      lossCoords4;
    vector<unique_ptr<NeuralLayerInterface>>
      layers1, layers2, layers3, layers4;
    auto ds = make_unique<UnseparatedCSVDataset>(
        "data/task1.txt", prop);
    vector<vector<pair<vector<float>,vector<float>>>> Xtr;
    vector<pair<vector<float>,vector<float>>> Xte;
    Xtr = ds->getTrainingExamples(B);
    Xte = ds->getTestingExamples();


    auto sig1 = make_unique<SigmoidLayer>(F,h);
    auto soft1 = make_unique<SoftMaxLayer>(h,C);
    layers1.push_back(move(sig1));
    layers1.push_back(move(soft1));
    auto nn1 = make_unique<NeuralNetwork>(move(layers1));
    nn1->train(Xtr,lr1,EPOCHS,loss1,true);
    for (int i=0; i<loss1.size(); i++) maxi1=max(maxi1,loss1[i]);
    acc1 = 100*nn1->test(Xte);
    for (int j=0; j<loss1.size(); j++)
      lossCoords1.push_back({(float)j,loss1[j]});

    auto sig2 = make_unique<SigmoidLayer>(F,h);
    auto soft2 = make_unique<SoftMaxLayer>(h,C);
    layers2.push_back(move(sig2));
    layers2.push_back(move(soft2));
    auto nn2 = make_unique<NeuralNetwork>(move(layers2));
    nn2->train(Xtr,lr2,EPOCHS,loss2,true);
    for (int i=0; i<loss2.size(); i++) maxi2=max(maxi2,loss2[i]);
    acc2 = 100*nn2->test(Xte);
    for (int j=0; j<loss2.size(); j++)
      lossCoords2.push_back({(float)j,loss2[j]});

    auto sig3 = make_unique<SigmoidLayer>(F,h);
    auto soft3 = make_unique<SoftMaxLayer>(h,C);
    layers3.push_back(move(sig3));
    layers3.push_back(move(soft3));
    auto nn3 = make_unique<NeuralNetwork>(move(layers3));
    nn3->train(Xtr,lr3,EPOCHS,loss3,true);
    for (int i=0; i<loss3.size(); i++) maxi3=max(maxi3,loss3[i]);
    acc3 = 100*nn3->test(Xte);
    for (int j=0; j<loss3.size(); j++)
      lossCoords3.push_back({(float)j,loss3[j]});

    auto sig4 = make_unique<SigmoidLayer>(F,h);
    auto soft4 = make_unique<SoftMaxLayer>(h,C);
    layers4.push_back(move(sig4));
    layers4.push_back(move(soft4));
    auto nn4 = make_unique<NeuralNetwork>(move(layers4));
    nn4->train(Xtr,lr4,EPOCHS,loss4,true);
    for (int i=0; i<loss4.size(); i++) maxi4=max(maxi4,loss4[i]);
    acc4 = 100*nn4->test(Xte);
    for (int j=0; j<loss4.size(); j++)
      lossCoords4.push_back({(float)j,loss4[j]});

    stringstream sslr, ssac; 
    sslr.str(string()); ssac.str(string());
    sslr << scientific << setprecision(1.f);
    ssac << fixed << setprecision(1.f);
    maxi = max(maxi1, maxi2); maxi=max(maxi,maxi3);
    maxi = max(maxi,maxi4);
    unique_ptr<Plotter> lossPlotter(new Plotter());

      shared_ptr<Plot> lossChart1(new Plot());
      lossChart1->addPlotData(lossCoords1,Scalar(255,0,0));
      lossChart1->setTicks(5,4);
      lossChart1->setXLabel("Train Iteration");
      lossChart1->setYLabel("Loss");
      ssac << acc1; sslr << lr1;
      lossChart1->setTitle(string("Lr: ")+
          sslr.str()+string(" (Test Acc: ")+
          ssac.str()+string("%)"));
      sslr.str(string()); ssac.str(string());
      lossChart1->setLimits({0,(float)loss1.size(),0,maxi});
      lossPlotter->addPlot(lossChart1);

      shared_ptr<Plot> lossChart2(new Plot());
      lossChart2->addPlotData(lossCoords2,Scalar(255,0,0));
      lossChart2->setTicks(5,4);
      lossChart2->setXLabel("Train Iteration");
      lossChart2->setYLabel("Loss");
      ssac << acc2; sslr << lr2;
      lossChart2->setTitle(string("Lr: ")+
          sslr.str()+string(" (Test Acc: ")+
          ssac.str()+string("%)"));
      sslr.str(string()); ssac.str(string());
      lossChart2->setLimits({0,(float)loss2.size(),0,maxi});
      lossPlotter->addPlot(lossChart2);

      shared_ptr<Plot> lossChart3(new Plot());
      lossChart3->addPlotData(lossCoords3,Scalar(255,0,0));
      lossChart3->setTicks(5,4);
      lossChart3->setXLabel("Train Iteration");
      lossChart3->setYLabel("Loss");
      ssac << acc3; sslr << lr3;
      lossChart3->setTitle(string("Lr: ")+
          sslr.str()+string(" (Test Acc: ")+
          ssac.str()+string("%)"));
      sslr.str(string()); ssac.str(string());
      lossChart3->setLimits({0,(float)loss3.size(),0,maxi});
      lossPlotter->addPlot(lossChart3);

      shared_ptr<Plot> lossChart4(new Plot());
      lossChart4->addPlotData(lossCoords4,Scalar(255,0,0));
      lossChart4->setTicks(5,4);
      lossChart4->setXLabel("Train Iteration");
      lossChart4->setYLabel("Loss");
      ssac << acc4; sslr << lr4;
      lossChart4->setTitle(string("Lr: ")+
          sslr.str()+string(" (Test Acc: ")+
          ssac.str()+string("%)"));
      sslr.str(string()); ssac.str(string());
      lossChart4->setLimits({0,(float)loss4.size(),0,maxi});
      lossPlotter->addPlot(lossChart4);

    lossPlotter->plot(
        string("Loss vs Iteration for different Learning rates ")+
        string("(algorithm: SGD, config: [10,sig:3,soft3], ")+
        string("Batch Size:10, EPOCHS: 100)"));
  }

  { // Show shuffle effect

    int F=10,C=3,h=3,EPOCHS=100,B=10;
    float lr=0.25, maxi, acc1, acc2, maxi1, maxi2;
    maxi1=maxi2=0.0;
    vector<float> loss1,loss2;
    vector<vector<float>> lossCoords1, lossCoords2;
    vector<unique_ptr<NeuralLayerInterface>>
      layers1, layers2;
    auto ds = make_unique<UnseparatedCSVDataset>(
        "data/task1.txt", prop);
    vector<vector<pair<vector<float>,vector<float>>>> Xtr1,Xtr2;
    vector<pair<vector<float>,vector<float>>> Xte;
    Xtr2 = ds->getTrainingExamples(B,false);
    Xtr1 = ds->getTrainingExamples(B);
    Xte = ds->getTestingExamples();


    auto sig1 = make_unique<SigmoidLayer>(F,h);
    auto soft1 = make_unique<SoftMaxLayer>(h,C);
    layers1.push_back(move(sig1));
    layers1.push_back(move(soft1));
    auto nn1 = make_unique<NeuralNetwork>(move(layers1));
    nn1->train(Xtr1,lr,EPOCHS,loss1,true);
    for (int i=0; i<loss1.size(); i++) maxi1=max(maxi1,loss1[i]);
    acc1 = 100*nn1->test(Xte);
    for (int j=0; j<loss1.size(); j++)
      lossCoords1.push_back({(float)j,loss1[j]});

    auto sig2 = make_unique<SigmoidLayer>(F,h);
    auto soft2 = make_unique<SoftMaxLayer>(h,C);
    layers2.push_back(move(sig2));
    layers2.push_back(move(soft2));
    auto nn2 = make_unique<NeuralNetwork>(move(layers2));
    nn2->train(Xtr2,lr,EPOCHS,loss2/*,false*/);
    for (int i=0; i<loss2.size(); i++) maxi2=max(maxi2,loss2[i]);
    acc2 = 100*nn2->test(Xte);
    for (int j=0; j<loss2.size(); j++)
      lossCoords2.push_back({(float)j,loss2[j]});

    stringstream ssac; 
    ssac.str(string());
    ssac << fixed << setprecision(1.f);
    maxi = max(maxi1, maxi2);
    unique_ptr<Plotter> lossPlotter(new Plotter());

      shared_ptr<Plot> lossChart1(new Plot());
      lossChart1->addPlotData(lossCoords1,Scalar(255,0,0));
      lossChart1->setTicks(5,4);
      lossChart1->setXLabel("Train Iteration");
      lossChart1->setYLabel("Loss");
      ssac << acc1;
      lossChart1->setTitle(string("Shuffled ")+
          string("(Test Acc: ")+
          ssac.str()+string("%)"));
      ssac.str(string());
      lossChart1->setLimits({0,(float)loss1.size(),0,maxi});
      lossPlotter->addPlot(lossChart1);

      shared_ptr<Plot> lossChart2(new Plot());
      lossChart2->addPlotData(lossCoords2,Scalar(255,0,0));
      lossChart2->setTicks(5,4);
      lossChart2->setXLabel("Train Iteration");
      lossChart2->setYLabel("Loss");
      ssac << acc2;
      lossChart2->setTitle(string("Not Shuffled ")+
          string("(Test Acc: ")+
          ssac.str()+string("%)"));
      ssac.str(string());
      lossChart2->setLimits({0,(float)loss2.size(),0,maxi});
      lossPlotter->addPlot(lossChart2);

    lossPlotter->plot(
        string("Loss vs Iteration for different Learning rates ")+
        string("(algorithm: SGD, config: [10,sig:3,soft3], ")+
        string("lr:0.25, Batch Size:10, EPOCHS: 100)"));
  }
}


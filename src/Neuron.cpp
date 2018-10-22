#include <Neuron.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

static default_random_engine gen;
static void printVec(const vector<float> &x) {
  for (float f: x) cout << f << ' ';
  cout << endl;
}

Neuron::Neuron(const int inputs, function<float(float)> act,
    function<float(float)> dact):
  _weights(vector<float>(inputs,0)),
  _wgrads(vector<float>(inputs,0)),
  _bgrad(0), _s(0),
  _x(vector<float>(inputs)),
  _bias(0), _act(act), _dact(dact) {}

Neuron::Neuron(const int inputs, const float std,
    function<float(float)> act, function<float(float)> dact):
  _weights(vector<float>(inputs,0)),
  _wgrads(vector<float>(inputs,0)),
  _bgrad(0), _s(0),
  _x(vector<float>(inputs)),
  _bias(0), _act(act), _dact(dact) {
  
    normal_distribution<float> ndist(0.f, std);
    for (int i=0; i<_weights.size(); i++)
      _weights[i] = ndist(gen);
  }

Neuron::Neuron(const vector<float> &weights, float bias,
    function<float(float)> act, function<float(float)> dact):
  _weights(weights), 
  _wgrads(vector<float>(weights.size(),0)),
  _bgrad(0), _s(0),
  _x(vector<float>(weights.size())),
  _bias(bias), _act(act), _dact(dact) {}

float Neuron::forward(const vector<float> &x) {
  _s = 0; _x = x;
  for (int i=0; i<_weights.size(); i++)
    _s += _weights[i]*x[i];
  return _act(_s+_bias);
}

vector<float> Neuron::backward(float e) {
  vector<float> ye(_x.size());
  float delta = e*_dact(_s); _bgrad = delta;
  for (int i=0, s=_x.size(); i<s; i++) {
    _wgrads[i] += delta*_x[i];
    ye[i] = delta*_weights[i];
  }
  return ye;
}

void Neuron::step(float lr) {
  for (int i=0; i<_weights.size(); i++)
    _weights[i] -= lr*_wgrads[i];
  _bias -= lr*_bgrad;

  _wgrads = vector<float>(_wgrads.size(),0);
  _bgrad = 0;
}

void Neuron::train(const vector<vector<float>> &X,
    const vector<float> &Y, float lr, int EPOCHS,
    vector<float> &loss) {

  float y_pred, error;
  for (int ep=0; ep<EPOCHS; ep++) {
    for (int i=0; i<X.size(); i++) {
      y_pred = forward(X[i]);
      error = y_pred-Y[i];
      backward(error);
      loss.push_back(pow(error,2));
      step(lr);
    }
  }
}
void Neuron::train(
    const vector<vector<pair<vector<float>,float>>> &X,
    float lr, int EPOCHS, vector<float> &loss) {

  float y_pred, error, e;
  for (int ep=0; ep<EPOCHS; ep++) {
    for (int i=0; i<X.size(); i++) {
      error = 0;
      for (int j=0; j<X[i].size(); j++) {
        y_pred = forward(X[i][j].first);
        e = y_pred-X[i][j].second; error += e;
        backward(e);
      }
      loss.push_back(pow(error/X[i].size(),2));
      step(lr/X[i].size());
    }
  }
}

float Neuron::test(const vector<vector<float>> &X,
    const vector<float> &Y, bool debug, string funcName) {

  float success=0, tot=X.size(), y_pred;
  for (int i=0; i<X.size(); i++) {
    y_pred = forward(X[i]);
    if (y_pred==Y[i]) {
      success++;
    } else {
      if (debug) {
        cout << funcName;
        cout << "Input: " <<'('<<X[i][0]<<','
          <<X[i][1]<<") ";
        cout << "Predicted: " << y_pred << " ";
        cout << "True: " << Y[i] << endl;
      }
    }
  }
  return success/tot;
}
float Neuron::test(const vector<pair<
    vector<float>,float>> &X,
    bool debug, string funcName) {

  float success=0, tot=X.size(), y_pred;
  for (int i=0; i<X.size(); i++) {
    y_pred = forward(X[i].first);
    if (y_pred==X[i].second) {
      success++;
    } else {
      if (debug) {
        cout << funcName;
        cout << "Input: " <<'('<<X[i].first[0]<<','
          <<X[i].first[1]<<") ";
        cout << "Predicted: " << y_pred << " ";
        cout << "True: " << X[i].second << endl;
      }
    }
  }
  return success/tot;
}
pair<vector<float>,float> Neuron::getWeights() {

  pair<vector<float>,float> weights_bss;
  for (int i=0, lsize=_weights.size(); i<lsize; i++)
    weights_bss.first.push_back(_weights[i]);
  weights_bss.second = _bias;

  return weights_bss;
}
void Neuron::setWeights(
    const pair<vector<float>,float> &weights_bss) {
  for (int i=0, wsize=_weights.size(); i<wsize; i++)
    _weights[i] = weights_bss.first[i];
  _bias = weights_bss.second;
}

static function<float(float)> sigmoidAct =
  [](float s)->float{return 1./(1+exp(-s));};
static function<float(float)> sigmoidClassifAct =
  [](float s)->float{return sigmoidAct(s)>0.5? 1.0:0.0;};
static function<float(float)> dsigmoidAct =
  [](float s)->float{return sigmoidAct(s)*(1.f-sigmoidAct(s));};
SigmoidNeuron::SigmoidNeuron(const int inputs, bool classif):
  Neuron(inputs, classif? sigmoidClassifAct:sigmoidAct,
      dsigmoidAct) {}
SigmoidNeuron::SigmoidNeuron(const int inputs, const float std):
  Neuron(inputs,std,sigmoidAct,dsigmoidAct) {}
SigmoidNeuron::SigmoidNeuron(
    const vector<float> &weights, float bias):
  Neuron(weights,bias,sigmoidClassifAct,dsigmoidAct) {}
TwoInputSigmoidNeuron::TwoInputSigmoidNeuron():
  SigmoidNeuron(2) {}
TwoInputSigmoidNeuron::TwoInputSigmoidNeuron(
    float w1,float w2,float b): 
  SigmoidNeuron(vector<float>{w1,w2}, b) {}
ANDSigmoidNeuron::ANDSigmoidNeuron():
  TwoInputSigmoidNeuron(0.5,0.5,-0.5) {}
ORSigmoidNeuron::ORSigmoidNeuron():
  TwoInputSigmoidNeuron(0.5,0.5,0.0) {}
NANDSigmoidNeuron::NANDSigmoidNeuron():
  TwoInputSigmoidNeuron(-0.5,-0.5,1.0) {}

static function<float(float)> reluAct =
  [](float s)->float{return s>0.f? s:0.f;};
static function<float(float)> dreluAct =
  [](float s)->float{return s>0.f? 1.f:0.f;};
ReluNeuron::ReluNeuron(const int inputs, const float std):
  Neuron(inputs,std,reluAct,dreluAct) {}
ReluNeuron::ReluNeuron(
    const vector<float> &weights, float bias):
  Neuron(weights,bias,reluAct,dreluAct) {}

static function<float(float)> idenAct =
  [](float s)->float{return s;};
static function<float(float)> didenAct =
  [](float s)->float{return 1.0;};
SoftMaxNeuron::SoftMaxNeuron(const int inputs, const float std):
  Neuron(inputs,std,idenAct,didenAct) {}
SoftMaxNeuron::SoftMaxNeuron(
    const vector<float> &weights, float bias):
  Neuron(weights,bias,idenAct,didenAct) {}

static function<float(float)> thresholdClassifAct =
  [](float s)->float{return s>0? 1.0:0.0;};
static function<float(float)> dthresholdClassifAct =
  [](float s)->float{return 1.0;};
Perceptron::Perceptron(int inputs):
  Neuron(inputs,thresholdClassifAct,dthresholdClassifAct) {}
Perceptron::Perceptron(const vector<float> &weights, float bias):
  Neuron(weights,bias,thresholdClassifAct,dthresholdClassifAct) {}
TwoInputPerceptron::TwoInputPerceptron(): Perceptron(2) {}
TwoInputPerceptron::TwoInputPerceptron(
    float w1,float w2,float b): 
  Perceptron(vector<float>{w1,w2}, b) {}
ANDPerceptron::ANDPerceptron():
  TwoInputPerceptron(0.5,0.5,-0.5) {}
ORPerceptron::ORPerceptron():
  TwoInputPerceptron(0.5,0.5,0.0) {}
NANDPerceptron::NANDPerceptron():
  TwoInputPerceptron(-0.5,-0.5,1.0) {}


NeuronXORGate::NeuronXORGate(unique_ptr<Neuron> nandNeu,
    unique_ptr<Neuron> orNeu, unique_ptr<Neuron> adderNeu): 
  _nandNeu(move(nandNeu)),_orNeu(move(orNeu)),
  _adderNeu(move(adderNeu)) {}
float NeuronXORGate::test(const vector<vector<float>> &X,
    const vector<float> &Y, bool debug) {

  float success=0, tot=X.size(), nand_out, or_out, xor_out;
  for (int i=0; i<X.size(); i++) {
    nand_out = _nandNeu->forward(X[i]);
    or_out = _orNeu->forward(X[i]);
    xor_out = _adderNeu->forward(vector<float>{nand_out,or_out});
    if (xor_out==Y[i]) {
      success++;
    } else {
      if (debug) {
        cout << "XOR Test ";
        cout << "Input: " <<'('<<X[i][0]<<','<<X[i][1]<<") ";
        cout << "Predicted: " << xor_out << " ";
        cout << "True: " << Y[i] << endl;
      }
    }
  }
  return success/tot;
}
SigmoidNeuronXORGate::SigmoidNeuronXORGate():
  NeuronXORGate(make_unique<NANDSigmoidNeuron>(),
    make_unique<ORSigmoidNeuron>(),
    make_unique<TwoInputSigmoidNeuron>(1.0,1.0,-1.0)) {}
PerceptronXORGate::PerceptronXORGate():
  NeuronXORGate(make_unique<NANDPerceptron>(),
    make_unique<ORPerceptron>(),
    make_unique<TwoInputPerceptron>(1.0,1.0,-1.0)) {}

template <class T> NeuralLayer<T>::NeuralLayer(const int neurons,
    const int inputs, function<float(float)> act,
    function<float(float)> dact) {

  for (int i=0; i<neurons; i++) {
    auto neuron = make_unique<T>(inputs, act, dact);
    _neurons.push_back(move(neuron));
  }
}
template <class T> vector<float> NeuralLayer<T>::forward(
    const vector<float> &x) {
  
  vector<float> y(_neurons.size());
  for (int i=0, size=_neurons.size(); i<size; i++)
    y[i] = _neurons[i]->forward(x);
  return y;
}
static vector<float> sumfv(vector<float> &v1,
 const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      plus<float>()); return v1;
}
template <class T> vector<float> NeuralLayer<T>::backward(
    const vector<float> &xe) {
  vector<float> ye = _neurons[0]->backward(xe[0]);
  for (int i=1, size=_neurons.size(); i<size; i++)
    sumfv(ye,_neurons[i]->backward(xe[i]));
  return ye;
}
template <class T> void NeuralLayer<T>::step(float lr) {
  for (int i=0, size=_neurons.size(); i<size; i++)
    _neurons[i]->step(lr);
}
template <class T> NeuralLayer<T>::NeuralLayer(const int inputs,
    const int neurons) {

  float std = sqrt(2.f/(inputs+neurons));
  for (int i=0; i<neurons; i++) {
    auto neuron = make_unique<T>(inputs,std);
    _neurons.push_back(move(neuron));
  }
}
template <class T> vector<pair<vector<float>,float>> 
  NeuralLayer<T>::getWeights() {

  vector<pair<vector<float>,float>> weights_bss;
  for (int i=0, lsize=_neurons.size(); i<lsize; i++)
    weights_bss.push_back(_neurons[i]->getWeights());

  return weights_bss;
}
template <class T> void NeuralLayer<T>::setWeights(
    const vector<pair<vector<float>,float>> &weights_bss) {
  for (int i=0, nsize=_neurons.size(); i<nsize; i++) {
    _neurons[i]->setWeights(weights_bss[i]);
  }
}
SigmoidLayer::SigmoidLayer(const int inputs, const int neurons):
  NeuralLayer<SigmoidNeuron>(inputs,neurons) {}
ReluLayer::ReluLayer(const int inputs, const int neurons):
  NeuralLayer<ReluNeuron>(inputs,neurons) {}
SoftMaxLayer::SoftMaxLayer(const int inputs, const int neurons):
  NeuralLayer<SoftMaxNeuron>(inputs,neurons) {}
static vector<float> softmaxfv(vector<float> &y_pred) {
  float maxComp = y_pred[0], den=0.f;
  for (int i=1, ysize=y_pred.size(); i<ysize; i++)
    maxComp = max(maxComp,y_pred[i]);
  for (int i=0, ysize=y_pred.size(); i<ysize; i++) {
    y_pred[i] = exp(y_pred[i]-maxComp);
    den += y_pred[i];
  }
  for (int i=0, ysize=y_pred.size(); i<ysize; i++)
    y_pred[i] = y_pred[i]/den;
  return y_pred;
}
vector<float> SoftMaxLayer::forward(
    const vector<float> &x) {
  vector<float> y_pred =
    NeuralLayer<SoftMaxNeuron>::forward(x);
  y_pred = softmaxfv(y_pred);
  return y_pred;
}

NeuralNetwork::NeuralNetwork(
    vector<unique_ptr<NeuralLayerInterface>> layers):
  _layers(move(layers)) {}
static vector<float> subfv(vector<float> &v1,
 const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      minus<float>()); return v1;
}
static float cross_entropyfv(const vector<float> &y,
 const vector<float> &y_pred){
  float loss = 0;
  for (int i=0; i<y.size(); i++)
    loss -= y[i]*log(y_pred[i]);
  return loss;
}
vector<float> NeuralNetwork::forward(const vector<float> &x) {
  
  vector<float> y_pred = x;
  for (int j=0, lsize=_layers.size(); j<lsize; j++)
    y_pred = _layers[j]->forward(y_pred);
  return y_pred;
}
vector<float> NeuralNetwork::backward(vector<float> error) {
  
  for (int j=_layers.size()-1; j>=0; j--)
    error = _layers[j]->backward(error);

  return error;
}
void NeuralNetwork::step(float lr) {
  
  for (int j=_layers.size()-1; j>=0; j--)
    _layers[j]->step(lr);
}
void NeuralNetwork::train(const vector<vector<float>> &X,
    const vector<vector<float>> &Y, float lr, int EPOCHS,
    vector<float> &loss, bool shuff) {
  
  vector<float> y_pred;
  for (int e=0; e<EPOCHS; e++) {
    //if (shuff) shuffle(gen);
    for (int i=0, xsize=X.size(); i<xsize; i++) {
      y_pred = forward(X[i]);
      loss.push_back(cross_entropyfv(Y[i], y_pred));
      backward(subfv(y_pred, Y[i]));
      step(lr);
    }
  }
}
void NeuralNetwork::train(
    vector<vector<pair<vector<float>,vector<float>>>> &X,
    float lr, int EPOCHS, vector<float> &loss, bool shuff) {
  
  vector<float> y_pred;
  for (int e=0; e<EPOCHS; e++) {
    if (shuff) shuffle(X.begin(),X.end(),gen);
    for (int i=0, xsize=X.size(); i<xsize; i++) {
      for (int j=0, bsize=X[i].size(); j<bsize; j++) { 
        y_pred = forward(X[i][j].first);
        loss.push_back(cross_entropyfv(X[i][j].second, y_pred));
        backward(subfv(y_pred, X[i][j].second));
      }
      step(lr/X[i].size());
    }
  }
}
static int maxIdxfv(const vector<float> &v){
  return distance(v.begin(),max_element(v.begin(),v.end()));
}
float NeuralNetwork::test(const vector<pair<
    vector<float>,vector<float>>> &X) {

  float success, tot=X.size();
  vector<float> y_pred;
  for (int i=0; i<tot; i++) {
    y_pred = X[i].first;
    for (int j=0, lsize=_layers.size(); j<lsize; j++)
      y_pred = _layers[j]->forward(y_pred);

    if (maxIdxfv(y_pred)==maxIdxfv(X[i].second))
      success++;
  }

  return success/tot;
}

vector<vector<pair<vector<float>,float>>> 
  NeuralNetwork::getWeights() {

  vector<vector<pair<vector<float>,float>>> weights_bss;
  for (int i=0, lsize=_layers.size(); i<lsize; i++)
    weights_bss.push_back(_layers[i]->getWeights());

  return weights_bss;
}

void NeuralNetwork::setWeights(const vector<vector<pair<
    vector<float>,float>>> &weights_bss) {

  for (int i=0, lsize=_layers.size(); i<lsize; i++) {
    _layers[i]->setWeights(weights_bss[i]);
  }
}

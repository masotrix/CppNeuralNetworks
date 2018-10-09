#include <Neuron.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>
using namespace std;

Neuron::Neuron(int inputs, function<float(float)> act):
  _weights(vector<float>(inputs,0)),
  _wgrads(vector<float>(inputs)),
  _x(vector<float>(inputs)),
  _bias(0), _act(act) {}

Neuron::Neuron(const vector<float> &weights, float bias,
    function<float(float)> act):
  _weights(weights), 
  _wgrads(vector<float>(weights.size())),
  _x(vector<float>(weights.size())),
  _bias(bias), _act(act) {}

float Neuron::forward(const vector<float> &x) {
  float output = 0; _x = x;
  for (int i=0; i<_weights.size(); i++)
    output += _weights[i]*x[i];
  if (output+_bias>0) return 1.0;
  else return 0.0;
}

void Neuron::backward(float y, float y_pred) {
  float diff = y_pred-y;
  for (int i=0; i<_weights.size(); i++)
    _wgrads[i] = _x[i]*diff;
  _bgrad = diff;
}

void Neuron::step(float lr) {
  for (int i=0; i<_weights.size(); i++)
    _weights[i] -= lr*_wgrads[i];
  _bias -= _bgrad;
}

void Neuron::train(const vector<vector<float>> &X,
    const vector<float> &Y, float lr, int EPOCHS,
    vector<float> &loss) {

  float y_pred;
  for (int ep=0; ep<EPOCHS; ep++) {
    for (int i=0; i<X.size(); i++) {
      y_pred = forward(X[i]);
      backward(Y[i], y_pred);
      loss.push_back(pow(y_pred-Y[i],2));
      step(lr);
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
        cout << "Input: " <<'('<<X[i][0]<<','<<X[i][1]<<") ";
        cout << "Predicted: " << y_pred << " ";
        cout << "True: " << Y[i] << endl;
      }
    }
  }
  return success/tot;
}

static function<float(float)> sigmoidAct =
  [](float s)->float{return 1/(1+exp(-s));};
SigmoidNeuron::SigmoidNeuron(int inputs):
  Neuron(inputs,sigmoidAct) {}
SigmoidNeuron::SigmoidNeuron(
    const vector<float> &weights, float bias):
  Neuron(weights,bias,sigmoidAct) {}
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

static function<float(float)> signAct =
  [](float s)->float{return s>0? 1.0:0.0;};
Perceptron::Perceptron(int inputs): Neuron(inputs,signAct) {}
Perceptron::Perceptron(const vector<float> &weights, float bias):
  Neuron(weights,bias,signAct) {}
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


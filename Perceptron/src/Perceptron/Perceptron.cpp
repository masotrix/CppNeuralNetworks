#include <Perceptron.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>
using namespace std;

Perceptron::Perceptron(int inputs):
  _weights(vector<float>(inputs,0)),
  _wgrads(vector<float>(inputs)),
  _x(vector<float>(inputs)),
  _bias(0) {}

Perceptron::Perceptron(const vector<float> &weights, float bias):
  _weights(weights), 
  _wgrads(vector<float>(weights.size())),
  _x(vector<float>(weights.size())),
  _bias(bias) {}

float Perceptron::forward(const vector<float> &x) {
  float output = 0; _x = x;
  for (int i=0; i<_weights.size(); i++)
    output += _weights[i]*x[i];
  if (output+_bias>0) return 1.0;
  else return 0.0;
}

void Perceptron::backward(float y, float y_pred) {
  float diff = y_pred-y;
  for (int i=0; i<_weights.size(); i++)
    _wgrads[i] = _x[i]*diff;
  _bgrad = diff;
}

void Perceptron::step(float lr) {
  for (int i=0; i<_weights.size(); i++)
    _weights[i] -= lr*_wgrads[i];
  _bias -= _bgrad;
}

void Perceptron::train(const vector<vector<float>> &X,
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

float Perceptron::test(const vector<vector<float>> &X,
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

XORGate::XORGate(): 
  _nandPer(unique_ptr<NANDPerceptron>(new NANDPerceptron())),
  _orPer(unique_ptr<ORPerceptron>(new ORPerceptron())),
  _adderPer(unique_ptr<TwoInputPerceptron>(
        new TwoInputPerceptron(1.0,1.0,-1.0))) {}

float XORGate::test(const vector<vector<float>> &X,
    const vector<float> &Y, bool debug) {

  float success=0, tot=X.size(), nand_out, or_out, xor_out;
  for (int i=0; i<X.size(); i++) {
    nand_out = _nandPer->forward(X[i]);
    or_out = _orPer->forward(X[i]);
    xor_out = _adderPer->forward(vector<float>{nand_out,or_out});
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


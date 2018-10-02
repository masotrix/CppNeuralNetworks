#include <Perceptron.h>
#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

Perceptron::Perceptron(int inputs):
  _weights(vector<float>(inputs,0)),
  _wgrads(vector<float>(inputs)),
  _x(vector<float>(inputs)),
  _bias(0), _act([](float s) {return s;}) {}

Perceptron::Perceptron(const vector<float> &weights, float bias,
    const function<float(float)> &act):
  _weights(weights), 
  _wgrads(vector<float>(weights.size())),
  _x(vector<float>(weights.size())),
  _bias(bias), _act(act) {}

float Perceptron::forward(const vector<float> &x) {
  float output = 0; _x = x;
  for (int i=0; i<_weights.size(); i++)
    output += _weights[i]*x[i];
  return _act(output+_bias);
}

void Perceptron::backward(float y, float y_pred) {
  float diff = y_pred-y;
  for (int i=0; i<_weights.size(); i++)
    _wgrads[i] = _x[i]*diff;
  _bgrad = diff;
}

void Perceptron::step(float gamma) {
  for (int i=0; i<_weights.size(); i++)
    _weights[i] -= gamma*_wgrads[i];
  _bias -= _bgrad;
}

void Perceptron::train(const vector<vector<float>> &X,
    const vector<float> &Y, int EPOCHS, vector<float> &loss) {

  float y_pred;
  for (int ep=0; ep<EPOCHS; ep++) {
    for (int i=0; i<X.size(); i++) {
      y_pred = forward(X[i]);
      backward(Y[i], y_pred);
      loss.push_back(pow(y_pred-Y[i],2));
      step(0.00001);
    }
  }
}

float Perceptron::test(const vector<vector<float>> &X,
    const vector<float> &Y) {

  float success=0, tot=X.size(), y_pred;
  for (int i=0; i<X.size(); i++) {
    y_pred = forward(X[i]);
    cout << "Input: " <<'('<<X[i][0]<<','<<X[i][1]<<") ";
    cout << "Predicted: " << y_pred << " ";
    cout << "True: " << Y[i] << " -> ";
    if (y_pred==Y[i]) {
      success++;
      cout << "OK\n";
    } else {
      cout << "FAIL\n";
    }
  }
  return success/tot;
}

float Perceptron::testOp(
    const vector<float> &weights, float bias,
    const function<float(float)> &act,
    const function<float(float, float)> &op,
    const vector<vector<float>> &X) {

  vector<float> Y(X.size());
  for (int i=0, s=Y.size(); i<s; i++)
    Y[i]=op(X[i][0],X[i][1]);

  unique_ptr<Perceptron> per(new Perceptron(weights,bias,act));
  return per->test(X,Y);
}



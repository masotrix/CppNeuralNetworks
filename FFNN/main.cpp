#include <iomanip>
#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
using namespace std;

void printVec(const vector<float> &x) {
  for (float f: x) cout << f << ' ';
  cout << endl;
}

void printMat(const vector<vector<float>> &w) {
  for (const vector<float> &wi: w) {
    for (float f: wi) cout << f << ' ';
    cout << endl;
  }
}


class Layer {

  private:
    vector<vector<float>> _w, _wgrads;
    vector<float> _x, _u, _bias, _bgrads;
    function<float (float)> _act, _dact;

  public:
    Layer(int inputs, int outputs, function<float (float)> &act,
        function<float (float)> &dact) {

      _w = vector<vector<float>>(inputs,
          vector<float>(outputs,0));
      _wgrads = vector<vector<float>>(inputs,
          vector<float>(outputs));
      _x = vector<float>(inputs);
      _u = vector<float>(outputs);
      _bias = vector<float>(outputs,0);
      _bgrads = vector<float>(outputs);
      _act = act; _dact = dact;
    }

    vector<float> forward(vector<float> &x) {
      _x = x;
      vector<float> y(_u.size());
      for (int i=0, s1=_u.size(); i<s1; i++) {
        float wsum = 0;
        for (int j=0, s2=_x.size(); j<s2; j++)
          wsum += _x[j]*_w[j][i];
        _u[i] = wsum + _bias[i];
        y[i] = _act(_u[i]);
      }
      return y;
    }

    void backward(vector<float> &error,
        vector<float> &error_return) {

      vector<float> error_next(_u.size());
      vector<float> delta(_u.size());

      for (int j=0, s2=_u.size(); j<s2; j++)
        delta[j] = _dact(_u[j])*error[j];
      for (int j=0, s1=_w.size(); j<s1; j++)
        for (int k=0; k<_u.size(); k++)
          _wgrads[j][k] = error[j]*delta[k];
      for (int j=0, s2=_bias.size(); j<s2; j++)
        _bgrads[j] = delta[j];
      for (int j=0, s2=_w.size(); j<s2; j++)
        for (int k=0, s3=_u.size(); k<s3; k++)
          error_next[j] = delta[k]*_w[j][k];

      error_return = move(error_next);
    }

    void step(float nu) {
      for (int i=0, s1=_w.size(); i<s1; i++) {
        for (int j=0, s2=_u.size(); j<s2; j++)
          _w[i][j] -+ nu*_wgrads[i][j];
      }

      for (int j=0, s2=_u.size(); j<s2; j++)
        _bias[j] -= nu*_bgrads[j];
    }

    void loadWeights(const vector<vector<float>> &w,
       const vector<float> &b) {
      for (int i=0, s1=w.size(); i<s1; i++) {
        for (int j=0, s2=w[0].size(); j<s2; j++)
          _w[i][j] = w[i][j];
        _bias[i] = b[i];
      }
    } 
};


int main() {
  
  function<float (float)> relu = [](float v){return max(0.f,v);},
    drelu = [](float v) {return v>0?1:0;};

  Layer *layer1 = new Layer(2,2,drelu,drelu);
  layer1->loadWeights({{1,1},{1,1}},{-1.5, -0.5});
  Layer *layer2 = new Layer(2,1,drelu,drelu);
  layer2->loadWeights({{-2},{1}},{-0.5});

  vector<vector<float>> examples = {{0,0},{1,0},{0,1},{1,1}};
  cout << "And   Or\n";
  cout << fixed << setprecision(2);
  for (int i=0; i<examples.size(); i++) {
    vector<float> out = layer1->forward(examples[i]);
    for (int j=0; j<out.size(); j++)
      cout << out[j] << "  ";
    cout << endl;
  }
  cout << "XOR\n";
  for (int i=0; i<examples.size(); i++) {
    vector<float> out1 = layer1->forward(examples[i]);
    vector<float> out2 = layer2->forward(out1);
    for (int j=0; j<out2.size(); j++)
      cout << out2[j] << "  ";
    cout << endl;
  }

  delete layer1;
  delete layer2;
}

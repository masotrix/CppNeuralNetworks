#include <memory>
#include <Neuron.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

/**
 * Class that encapsulates NeuralNetwork & NeuralLayer 
 * Unit Testing
 */
class TestNeuralNetwork {

  private:
    /* Activation functions and their derivatives */
    TestNeuralNetwork() = delete;
    static vector<float> _sumfv(vector<float>&,
      const vector<float>&);
    static function<float(float)> _sigmoidAct;
    static function<float(float)> _dsigmoidAct;
    static vector<float> _softmaxfv(const vector<float> &y_pred);

  public:
    // Test Sigmoid Neuron Forward
    static void testSigmoidNeuronForward() {

      // Data
        vector<float> x = {1.0,2.0,3.0}; float y, y_pred, sb=0;
        pair<vector<float>,float>  w = {
          {1.0, 2.0, 3.0},1.0};

      { // Test Equality
        for (int i=0; i<w.first.size(); i++) sb+=x[i]*w.first[i];
        sb += w.second; y = _sigmoidAct(sb);
        auto sig = make_unique<SigmoidNeuron>(3,false);
        sig->setWeights(w); y_pred = sig->forward(x);

        if (y_pred!=y) {
          cout << "Error in Sigmoid Neuron Forward Output\n"; 
          exit(1);
        }
      }
    }

    // Test Sigmoid Neuron Backward
    static void testSigmoidNeuronBackward() {

      // Data
      vector<float> x={1,2}, xe(x.size(),0), xe_pred;
      float ye = -0.7, y, y_pred, s, ds;
      pair<vector<float>,float>  w = {
        {1.0, 2.0, 3.0},1.0};

      { // Test Equality
        for (int i=0; i<w.first.size(); i++) s+=x[i]*w.first[i];
        ds = ye*_dsigmoidAct(s); xe[0]=ds*x[0]; xe[1]=ds*x[1]; 
        auto sig = make_unique<SigmoidNeuron>(3,false);
        sig->setWeights(w); y_pred = sig->forward(x);
        xe_pred = sig->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in Sigmoid Neuron Backward Output\n"; 
          exit(1);
        }
      }
    }

    // Test SoftMax Neuron Forward
    static void testSoftMaxNeuronForward() {

      // Data
        vector<float> x = {1.0,2.0,3.0}; float y, y_pred, sb=0;
        pair<vector<float>,float>  w = {
          {1.0, 2.0, 3.0},1.0};

      { // Test Equality
        for (int i=0; i<w.first.size(); i++) sb+=x[i]*w.first[i];
        sb += w.second; y = sb;
        auto soft = make_unique<SoftMaxNeuron>(3,false);
        soft->setWeights(w); y_pred = soft->forward(x);

        if (y_pred!=y) {
          cout << "Error in SoftMax Neuron Forward Output\n"; 
          exit(1);
        }
      }
    }

    // Test SoftMax Neuron Backward
    static void testSoftMaxNeuronBackward() {

      // Data
      vector<float> x={1,2}, xe(x.size(),0), xe_pred;
      float ye = -0.7, y, y_pred, ds;
      pair<vector<float>,float>  w = {
        {1.0, 2.0, 3.0},1.0};

      { // Test Equality
        ds = ye; xe[0]=ds*x[0]; xe[1]=ds*x[1]; 
        auto soft = make_unique<SoftMaxNeuron>(3,false);
        soft->setWeights(w); y_pred = soft->forward(x);
        xe_pred = soft->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in SoftMax Neuron Backward Output\n"; 
          exit(1);
        }
      }
    }

    // Test Sigmoid Layer Forward
    static void testSigmoidLayerForward() {
      
      // Data
        vector<float> x = {1.0,2.0,3.0}, y, y_pred; float y1,y2;
        vector<pair<vector<float>,float>>  w = {
          {{1.0, 2.0, 3.0},1.0},{{3.0,2.0,1.0},2.0}};
      //
       
      { // Test
        auto sig1 = make_unique<SigmoidNeuron>(3,false);
        auto sig2 = make_unique<SigmoidNeuron>(3,false);
        sig1->setWeights(w[0]); sig2->setWeights(w[1]); 
        auto sigl = make_unique<SigmoidLayer>(3,2);
        sigl->setWeights(w);

        y1 = sig1->forward(x); y2 = sig2->forward(x);
        y = {y1,y2};
        y_pred = sigl->forward(x);

        { // Forward
          if (y_pred!=y) {
            cout << "Error in Sigmoid Layer Forward Output\n"; 
            exit(1);
          }
        } //
      }
    }

    // Test Sigmoid Layer Backward
    static void testSigmoidLayerBackward() {
      
      // Data
        vector<float> ye = {-0.7,0.3,0.4}, x = {1, 2}, y_pred,
          xe(x.size(),0), xe_pred; float y;
        vector<pair<vector<float>,float>>  w = {
          {{1, 2},1},{{3,2},2},{{1,3},3}};
      //
      
       
      { // Test Equality
        auto sig1 = make_unique<SigmoidNeuron>(2,false);
        auto sig2 = make_unique<SigmoidNeuron>(2,false);
        auto sig3 = make_unique<SigmoidNeuron>(2,false);
        sig1->setWeights(w[0]); sig2->setWeights(w[1]);
        sig3->setWeights(w[2]);
        auto sigl = make_unique<SigmoidLayer>(2,3);
        sigl->setWeights(w);
        y = sig1->forward(x); y = sig2->forward(x);
        y = sig3->forward(x);
        y_pred = sigl->forward(x);

        xe = _sumfv(xe,sig1->backward(ye[0]));
        xe = _sumfv(xe,sig2->backward(ye[1]));
        xe = _sumfv(xe,sig3->backward(ye[2]));
        xe_pred = sigl->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in Sigmoid Layer Backward Output\n"; 
          exit(1);
        }
      }
    }

    // Test SoftMax Layer Forward
    static void testSoftMaxLayerForward() {
  
      // Data
        vector<float> x = {1.0,2.0,3.0}, y, y_pred; float y1,y2;
        vector<pair<vector<float>,float>>  w = {
          {{1.0, 2.0, 3.0},1.0},{{3.0,2.0,1.0},2.0}};
      //
       
      { // Test
        auto soft1 = make_unique<SoftMaxNeuron>(3,1.0);
        auto soft2 = make_unique<SoftMaxNeuron>(3,1.0);
        soft1->setWeights(w[0]); soft2->setWeights(w[1]); 
        auto softl = make_unique<SoftMaxLayer>(3,2);
        softl->setWeights(w);

        y1 = soft1->forward(x); y2 = soft2->forward(x);
        y = _softmaxfv({y1,y2});
        y_pred = softl->forward(x);

        { // Forward
          if (y_pred!=y) {
            cout << "Error in SoftMax Layer Forward Output\n"; 
            exit(1);
          }
        } //
      }
    }

    // Test SoftMax Layer Backward
    static void testSoftMaxLayerBackward() {
      
      // Data
        vector<float> ye = {-0.7,0.3,0.4}, x = {1, 2}, y_pred,
          xe(x.size(),0), xe_pred; float y;
        vector<pair<vector<float>,float>>  w = {
          {{1, 2},1},{{3,2},2},{{1,3},3}};
      //
      
       
      { // Test Equality
        auto soft1 = make_unique<SoftMaxNeuron>(2,false);
        auto soft2 = make_unique<SoftMaxNeuron>(2,false);
        auto soft3 = make_unique<SoftMaxNeuron>(2,false);
        soft1->setWeights(w[0]); soft2->setWeights(w[1]);
        soft3->setWeights(w[2]);
        auto softl = make_unique<SoftMaxLayer>(2,3);
        softl->setWeights(w);
        y = soft1->forward(x); y = soft2->forward(x);
        y = soft3->forward(x);
        y_pred = softl->forward(x);

        xe = _sumfv(xe,soft1->backward(ye[0]));
        xe = _sumfv(xe,soft2->backward(ye[1]));
        xe = _sumfv(xe,soft3->backward(ye[2]));
        xe_pred = softl->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in SoftMax Layer Backward Output\n"; 
          exit(1);
        }
      }
    }

    // Test Classification Network Forward
    static void testClassificationNetworkForward() {
      
      // Data
        vector<float> x = {1,2,3}, y, y_pred;
        vector<vector<pair<vector<float>,float>>>  w = {
        {{{1,2,3},1},{{3,2,1},2},{{1,3,2},3}},
        {{{4,5,6},1},{{6,5,4},2}}};
      //

      { // Test equality
        auto sig1 = make_unique<SigmoidLayer>(3,3);
        auto soft1 = make_unique<SoftMaxLayer>(3,2);
        sig1->setWeights(w[0]); soft1->setWeights(w[1]);
        y = sig1->forward(x); y = soft1->forward(y);
        vector<unique_ptr<NeuralLayerInterface>> layers;
        auto sig2 = make_unique<SigmoidLayer>(3,3);
        auto soft2 = make_unique<SoftMaxLayer>(3,2);
        layers.push_back(move(sig2));
        layers.push_back(move(soft2));
        auto nn = make_unique<NeuralNetwork>(move(layers));
        nn->setWeights(w);
        y_pred = nn->forward(x);

        if (y_pred!=y) {
          cout << "Error in Classification Network Forward\n";
          exit(1);
        }
      }
    }

    // Test Classification Network Backward
    static void testClassificationNetworkBackward() {

      // Data
        vector<float> ye = {1,2,3}, x = {1,2}, y, xe, xe_pred;
        vector<vector<pair<vector<float>,float>>>  w = {
        {{{1,2,3},1},{{3,2,1},2},{{1,3,2},3}},
        {{{4,5,6},1},{{6,5,4},2},{{4,6,5},3}}};
      //

      { // Test equality
        auto sig1 = make_unique<SigmoidLayer>(2,3);
        auto soft1 = make_unique<SoftMaxLayer>(3,3);
        sig1->setWeights(w[0]); soft1->setWeights(w[1]);
        vector<unique_ptr<NeuralLayerInterface>> layers;
        auto sig2 = make_unique<SigmoidLayer>(2,3);
        auto soft2 = make_unique<SoftMaxLayer>(3,3);
        layers.push_back(move(sig2));
        layers.push_back(move(soft2));
        auto nn = make_unique<NeuralNetwork>(move(layers));
        nn->setWeights(w);
        y = sig1->forward(x); y = soft1->forward(y);
        y = nn->forward(x);
        xe = soft1->backward(ye); xe = sig1->backward(xe);
        xe_pred = nn->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in Classification Network Backward\n";
          exit(1);
        }
      }
    }

    static void testRegressionNetworkForward() {

      // Data
        vector<float> x = {1,2,3}, y, y_pred;
        vector<vector<pair<vector<float>,float>>>  w = {
        {{{1,2,3},1},{{3,2,1},2},{{1,3,2},3}},
        {{{4,5,6},1},{{6,5,4},2}}};
      //

      { // Test equality
        auto sig11 = make_unique<SigmoidLayer>(3,3);
        auto sig12 = make_unique<SigmoidLayer>(3,2);
        sig11->setWeights(w[0]); sig12->setWeights(w[1]);
        y = sig11->forward(x); y = sig12->forward(y);
        vector<unique_ptr<NeuralLayerInterface>> layers;
        auto sig21 = make_unique<SigmoidLayer>(3,3);
        auto sig22 = make_unique<SigmoidLayer>(3,2);
        layers.push_back(move(sig21));
        layers.push_back(move(sig22));
        auto nn = make_unique<NeuralNetwork>(move(layers));
        nn->setWeights(w);
        y_pred = nn->forward(x);

        if (y_pred!=y) {
          cout << "Error in Regression Network Forward\n";
          exit(1);
        }
      }
    }

    static void testRegressionNetworkBackward() {

      // Data
        vector<float> ye = {1,2,3}, x = {1,2}, y, xe, xe_pred;
        vector<vector<pair<vector<float>,float>>>  w = {
        {{{1,2,3},1},{{3,2,1},2},{{1,3,2},3}},
        {{{4,5,6},1},{{6,5,4},2},{{4,6,5},3}}};
      //

      { // Test equality
        auto sig11 = make_unique<SigmoidLayer>(2,3);
        auto sig12 = make_unique<SigmoidLayer>(3,3);
        sig11->setWeights(w[0]); sig12->setWeights(w[1]);
        vector<unique_ptr<NeuralLayerInterface>> layers;
        auto sig21 = make_unique<SigmoidLayer>(2,3);
        auto sig22 = make_unique<SigmoidLayer>(3,3);
        layers.push_back(move(sig21));
        layers.push_back(move(sig22));
        auto nn = make_unique<NeuralNetwork>(move(layers));
        nn->setWeights(w);
        y = sig11->forward(x); y = sig12->forward(y);
        y = nn->forward(x);
        xe = sig12->backward(ye); xe = sig11->backward(xe);
        xe_pred = nn->backward(ye);

        if (xe_pred!=xe) {
          cout << "Error in Regression Network Backward\n";
          exit(1);
        }
      }
    }
};

vector<float> TestNeuralNetwork::_sumfv(vector<float> &v1,
 const vector<float> &v2){
  transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),
      plus<float>()); return v1;
}
function<float(float)> TestNeuralNetwork::_sigmoidAct =
  [](float s)->float{return 1./(1+exp(-s));};
function<float(float)> TestNeuralNetwork::_dsigmoidAct=
  [](float s)->float{return _sigmoidAct(s)*(1.f-_sigmoidAct(s));};
vector<float> TestNeuralNetwork::_softmaxfv(
    const vector<float> &s) {
  vector<float> y_pred(s);
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



int main() {

  // Test Sigmoid Neuron
  TestNeuralNetwork::testSigmoidNeuronForward();
  TestNeuralNetwork::testSigmoidNeuronBackward();
  
  // Test SoftMax Neuron
  TestNeuralNetwork::testSoftMaxNeuronForward();
  TestNeuralNetwork::testSoftMaxNeuronBackward();

  // Test Sigmoid Layer
  TestNeuralNetwork::testSigmoidLayerForward();
  TestNeuralNetwork::testSigmoidLayerBackward();

  // Test SoftMax Layer
  TestNeuralNetwork::testSoftMaxLayerForward();
  TestNeuralNetwork::testSoftMaxLayerBackward();

  // Test Classification Network
  TestNeuralNetwork::testClassificationNetworkForward();
  TestNeuralNetwork::testClassificationNetworkBackward();

  // Test Regression Network
  TestNeuralNetwork::testRegressionNetworkForward();
  TestNeuralNetwork::testRegressionNetworkBackward();

  // Everything OK Feedback
  cout << "All Network Unit Tests OK\n";
}

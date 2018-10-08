#include <vector>
#include <functional>
#include <memory>

class Perceptron {

  private:
    std::vector<float> _x, _weights, _wgrads;
    float _bias, _bgrad;

  public:
    Perceptron(int inputs);
    Perceptron(const std::vector<float> &, float);

    virtual float forward(const std::vector<float> &x);

    virtual void backward(float y, float y_pred);

    virtual void step(float lr);

    virtual void train(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, float lr, int EPOCHS,
        std::vector<float> &loss);

    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, bool debug=false,
        std::string funcName = std::string());
};

class TwoInputPerceptron: public Perceptron {

  public:
    TwoInputPerceptron();
    TwoInputPerceptron(float w1, float w2, float b);
};

class ANDPerceptron: public TwoInputPerceptron {
  public: ANDPerceptron();
};

class ORPerceptron: public TwoInputPerceptron {
  public: ORPerceptron();
};

class NANDPerceptron: public TwoInputPerceptron {
  public: NANDPerceptron();
};

class XORGate {

  private:
    std::unique_ptr<NANDPerceptron> _nandPer;
    std::unique_ptr<ORPerceptron> _orPer;
    std::unique_ptr<TwoInputPerceptron> _adderPer;
 
  public:
    XORGate();   
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, bool debug = false);
   
};

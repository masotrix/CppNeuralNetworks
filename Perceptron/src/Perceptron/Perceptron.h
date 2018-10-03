#include <vector>
#include <functional>
#include <memory>

class Perceptron {

  private:
    std::vector<float> _x, _weights, _wgrads;
    float _bias, _bgrad;
    std::function<float(float)> _act;

  public:
    Perceptron(int inputs);
    Perceptron(const std::vector<float> &, float,
        const std::function<float(float)>&act=
        [](float s)->float {return s;});

    virtual float forward(const std::vector<float> &x);

    virtual void backward(float y, float y_pred);

    virtual void step(float gamma);

    virtual void train(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, int EPOCHS,
        std::vector<float> &loss);

    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y);

    static float testOp(
      const std::vector<float> &weights, float bias,
      const std::function<float(float)> &act,
      const std::function<float(float, float)> &op,
      const std::vector<std::vector<float>> &X);
};

class TwoInputPerceptron: public Perceptron {

  public:
    TwoInputPerceptron();
    TwoInputPerceptron(float w1, float w2, float b,
        const std::function<float(float)>&act=
        [](float s)->float {return s;});
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

class SUMPerceptron: public TwoInputPerceptron {
  public: SUMPerceptron();
};

class XORGate {

  private:
    std::unique_ptr<NANDPerceptron> _nandPer;
    std::unique_ptr<ORPerceptron> _orPer;
    std::unique_ptr<TwoInputPerceptron> _adderPer;
 
  public:
    XORGate();   
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y);
   
};

#include <vector>
#include <functional>
#include <memory>

class Neuron {

  private:
    std::vector<float> _x, _weights, _wgrads;
    float _bias, _bgrad;
    std::function<float(float)> _act;

  public:
    explicit Neuron(int inputs, std::function<float(float)>);
    explicit Neuron(const std::vector<float> &, float,
        std::function<float(float)>);
    Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;
    ~Neuron() = default;

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

class SigmoidNeuron: public Neuron {
  public:
    explicit SigmoidNeuron(int inputs);
    explicit SigmoidNeuron(const std::vector<float> &, float);
    SigmoidNeuron(const SigmoidNeuron&) = delete;
    SigmoidNeuron& operator=(const SigmoidNeuron&) = delete;
    ~SigmoidNeuron() = default;
};
class TwoInputSigmoidNeuron: public SigmoidNeuron {
  public:
    explicit TwoInputSigmoidNeuron();
    explicit TwoInputSigmoidNeuron(float w1, float w2, float b);
    TwoInputSigmoidNeuron(const TwoInputSigmoidNeuron&) = delete;
    TwoInputSigmoidNeuron& operator=(
        const TwoInputSigmoidNeuron&) = delete;
    ~TwoInputSigmoidNeuron() = default;
};
class ANDSigmoidNeuron: public TwoInputSigmoidNeuron {
  public: ANDSigmoidNeuron();
};
class ORSigmoidNeuron: public TwoInputSigmoidNeuron {
  public: ORSigmoidNeuron();
};
class NANDSigmoidNeuron: public TwoInputSigmoidNeuron {
  public: NANDSigmoidNeuron();
};


class Perceptron: public Neuron {
  public:
    Perceptron(int inputs);
    Perceptron(const std::vector<float> &, float);
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


class NeuronXORGate {
  private:
    std::unique_ptr<Neuron> _nandNeu;
    std::unique_ptr<Neuron> _orNeu;
    std::unique_ptr<Neuron> _adderNeu;
  public:
    NeuronXORGate(std::unique_ptr<Neuron>,
        std::unique_ptr<Neuron>,std::unique_ptr<Neuron>);   
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, bool debug = false);
};
class SigmoidNeuronXORGate: public NeuronXORGate {
  public:
    SigmoidNeuronXORGate();   
};
class PerceptronXORGate: public NeuronXORGate {
  public:
    PerceptronXORGate();   
};

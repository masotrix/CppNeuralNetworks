#include <vector>
#include <functional>
#include <memory>

class Neuron {

  private:
    std::vector<float> _x, _weights, _wgrads;
    float _bias, _bgrad, _s;
    std::function<float(float)> _act, _dact;

  public:
    explicit Neuron(const int inputs, std::function<float(float)>,
        std::function<float(float)>);
    explicit Neuron(const int, float, std::function<float(float)>,
        std::function<float(float)>);
    explicit Neuron(const std::vector<float> &, float,
        std::function<float(float)>,std::function<float(float)>);
    Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;
    ~Neuron() = default;

    virtual float forward(const std::vector<float> &x);
    virtual void backward(float y, float y_pred);
    virtual std::vector<float> backout(float e);
    virtual void step(float lr);

    virtual void train(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, float lr, int EPOCHS,
        std::vector<float> &loss);
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, bool debug=false,
        std::string funcName = std::string());
    virtual void loadWeights(const std::vector<float>&);
};

class SigmoidNeuron: public Neuron {
  public:
    explicit SigmoidNeuron(const int inputs, bool classif=true);
    explicit SigmoidNeuron(const int inputs, const float);
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

class ReluNeuron: public Neuron {
  public:
    explicit ReluNeuron(const int inputs, const float);
    explicit ReluNeuron(const std::vector<float> &, float);
    ReluNeuron(const ReluNeuron&) = delete;
    ReluNeuron& operator=(const ReluNeuron&) = delete;
    ~ReluNeuron() = default;
};

class SoftMaxNeuron: public Neuron {
  public:
    explicit SoftMaxNeuron(const int inputs, const float);
    explicit SoftMaxNeuron(const std::vector<float> &, float);
    SoftMaxNeuron(const SoftMaxNeuron&) = delete;
    SoftMaxNeuron& operator=(const SoftMaxNeuron&) = delete;
    ~SoftMaxNeuron() = default;
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
  public: SigmoidNeuronXORGate();   
};
class PerceptronXORGate: public NeuronXORGate {
  public: PerceptronXORGate();   
};

class NeuralLayerInterface {
  public:
    explicit NeuralLayerInterface() {}
    NeuralLayerInterface(const NeuralLayerInterface&) = delete;
    NeuralLayerInterface& operator=(const NeuralLayerInterface&)
      = delete;
    ~NeuralLayerInterface() = default;

    virtual std::vector<float> forward(
        const std::vector<float> &x)= 0;
    virtual std::vector<float> backward(
        const std::vector<float> &xe) = 0;
    virtual void step(float lr) = 0;
    virtual void loadWeights(
        const std::vector<std::vector<float>>&)=0;
};

template <class T> class NeuralLayer:
  public NeuralLayerInterface {

  private:
    std::vector<std::unique_ptr<T>> _neurons;

  protected:
    NeuralLayer(const int neurons, const int inputs);
  
  public:
    explicit NeuralLayer(const int neurons, const int inputs,
        std::function<float(float)> act,
        std::function<float(float)> dact);
    NeuralLayer(const NeuralLayer&) = delete;
    NeuralLayer& operator=(const NeuralLayer&) = delete;
    ~NeuralLayer() = default;

    std::vector<float> forward(const std::vector<float> &x);
    std::vector<float> backward(const std::vector<float> &xe);
    void step(float lr);
    void loadWeights(const std::vector<std::vector<float>>&);
};
class SigmoidLayer: public NeuralLayer<SigmoidNeuron> {
  public: SigmoidLayer(const int inputs, const int neurons);
};
class ReluLayer: public NeuralLayer<ReluNeuron> {
  public: ReluLayer(const int inputs, const int neurons);
};
class SoftMaxLayer: public NeuralLayer<SoftMaxNeuron> {
  public: SoftMaxLayer(const int inputs, const int neurons);
  std::vector<float> forward(const std::vector<float> &x)
    override;
};


class NeuralNetwork {
  private:
    std::vector<std::unique_ptr<NeuralLayerInterface>> _layers;

  public:
    explicit NeuralNetwork(std::vector<std::unique_ptr<
        NeuralLayerInterface>>);
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    ~NeuralNetwork() = default;
    
    virtual std::vector<float>forward(const std::vector<float>&);
    virtual void backward(std::vector<float>,float);
    virtual void train(const std::vector<std::vector<float>> &X,
        const std::vector<std::vector<float>> &Y,
        float lr, int EPOCHS,std::vector<float> &loss);
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<std::vector<float>> &Y);
    virtual void loadWeights(
        const std::vector<std::vector<std::vector<float>>>&);
};

#include <vector>
#include <functional>
#include <memory>

/**
 *  Class representing an artificial neuron, the one has
 *  basic structure components (inputs, weights, bias,
 *  error gradients and activation function) and responsive
 *  behavior and learning behavior (forward, backward and step).
 *  It may set and retrieve its learned weights too.
 */
class Neuron {

  private:
    std::vector<float> _x, _weights, _wgrads;
    float _bias, _bgrad, _s;
    std::function<float(float)> _act, _dact;

  public:
    /**
     *  Basic constructor receiving number of inputs and
     *  activation function (and its derivative).
     */
    explicit Neuron(const int inputs, std::function<float(float)>,
        std::function<float(float)>);
    /**
     *  Similar to basic constructor, but receives standard
     *  deviation for initilizing its weights (with 0 centered
     *  normal distribution).
     */ 
    explicit Neuron(const int, float, std::function<float(float)>,
        std::function<float(float)>);
    /**
     *  Similar to basic constructur, but instead of receiving
     *  number of inputs, it receives its weights and bias.
     */
    explicit Neuron(const std::vector<float> &, float,
        std::function<float(float)>,std::function<float(float)>);
    Neuron(const Neuron&) = delete;
    Neuron& operator=(const Neuron&) = delete;
    ~Neuron() = default;

    /**
     *  Forward Neuron method thar receives example inputs
     */
    virtual float forward(const std::vector<float> &x);
    /**
     *  Backward Neuron method that receives example error
     */ 
    virtual std::vector<float> backward(float e);
    /**
     *  Step Neuron method that receives learning rate
     */ 
    virtual void step(float lr);
    virtual void train(
        const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, float lr,
        int EPOCHS, std::vector<float> &loss);
    /**
     *  Train Neuron method that receives input data as
     *  bataches of pairs of examples and true outputs,
     *  learning rate, epochs to be trained and a vector
     *  to store loss values obtained
     */
    virtual void train(
        const std::vector<std::vector<std::pair<
          std::vector<float>,float>>> &X,
        float lr, int EPOCHS, std::vector<float> &loss);
    virtual float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y,
        bool debug=false,std::string funcName = std::string());
    /**
     *  Test Neuron method that receives input data, if it is
     *  allowed to print debug info, and a string representing
     *  a function being tested. 
     */ 
    virtual float test(const std::vector<
        std::pair<std::vector<float>,float>> &X,
        bool debug=false,std::string funcName = std::string());
    /**
     *  Retrieve Weights method, as pair of weigths and the bias
     */
    virtual std::pair<std::vector<float>,float> getWeights();
    /**
     *  Set Weights method, as pair of weigths and the bias
     */
    virtual void setWeights(const std::pair<
        std::vector<float>,float>&);
};

/**
 *  Sigmoid Neuron, the one uses the sigmoid activation function
 *  and learns as such using its derivative inside backward
 */ 
class SigmoidNeuron: public Neuron {
  public:
    /**
     *  Basic constructor that receives number of inputs and
     *  specifies if used for classification or not (in which
     *  case, its activation function aplies a threshold too).
     */ 
    explicit SigmoidNeuron(const int inputs, bool classif=true);
    /**
     *  Same as basic constructor, but not classifies and
     *  receives standard deviation as corresponding
     *  Neuron constructor (the one uses behind scenes)
     */ 
    explicit SigmoidNeuron(const int inputs, const float);
    /**
     *  Same as Neuron set weight constructor,
     *  but sets inside sigmoid activation function
     */ 
    explicit SigmoidNeuron(const std::vector<float> &, float);
    SigmoidNeuron(const SigmoidNeuron&) = delete;
    SigmoidNeuron& operator=(const SigmoidNeuron&) = delete;
    ~SigmoidNeuron() = default;
};
/**
 *  SigmoidNeuron with two inputs
 */ 
class TwoInputSigmoidNeuron: public SigmoidNeuron {
  public:
    /**
     *  Basic constructor 'alias' for SigmoidNeuron(2)
     */ 
    explicit TwoInputSigmoidNeuron();
    /**
     *  Set weights constructor, 'alias'
     *  for SigmoidNeuron({w1,w2}, b)
     */ 
    explicit TwoInputSigmoidNeuron(float w1, float w2, float b);
    TwoInputSigmoidNeuron(const TwoInputSigmoidNeuron&) = delete;
    TwoInputSigmoidNeuron& operator=(
        const TwoInputSigmoidNeuron&) = delete;
    ~TwoInputSigmoidNeuron() = default;
};
/**
*  Two Input Sigmoid Neuron that sets its weights to
*  behave like an AND gate.
*/ 
class ANDSigmoidNeuron: public TwoInputSigmoidNeuron {
  /**
   *  Constructor that sets weights using
   *  TwoInputSigmoidNeuron(w1,w2,b) (with proper w1,w2,b)
   */ 
  public: ANDSigmoidNeuron();
};
/**
 *  Two Input Sigmoid Neurno that sets its weights to
 *  behave like an OR gate.
 */ 
class ORSigmoidNeuron: public TwoInputSigmoidNeuron {
  /**
   *  Constructor that sets weights using
   *  TwoInputSigmoidNeuron(w1,w2,b) (with proper w1,w2,b)
   */ 
  public: ORSigmoidNeuron();
};
/**
 *  TwoInputSigmoid Neurno that sets its weights to
 *  behave like a NAND gate.
 */ 
class NANDSigmoidNeuron: public TwoInputSigmoidNeuron {
  /**
   *  Constructor that sets weights using
   *  TwoInputSigmoidNeuron(w1,w2,b) (with proper w1,w2,b)
   */ 
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
    virtual std::vector<std::pair<std::vector<float>,float>>
      getWeights()=0;
    virtual void setWeights(
        const std::vector<std::pair<
        std::vector<float>,float>>&)=0;
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
    std::vector<std::pair<std::vector<float>,float>>getWeights();
    void setWeights(const std::vector<std::pair<
        std::vector<float>,float>>&);
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
    virtual std::vector<float>backward(std::vector<float>);
    virtual void step(float);
    virtual void train(
        const std::vector<std::vector<float>> &X,
        const std::vector<std::vector<float>> &Y,
        float lr, int EPOCHS,std::vector<float> &loss,
        bool shuffle=true);
    virtual void train(
        std::vector<std::vector<std::pair<
        std::vector<float>,std::vector<float>>>> &X,
        float lr, int EPOCHS,std::vector<float> &loss,
        bool shuffle=true);
    virtual float test(const std::vector<std::pair<
        std::vector<float>,std::vector<float>>> &X);
    virtual std::vector<std::vector<std::pair<
        std::vector<float>,float>>> getWeights();
    virtual void setWeights(
        const std::vector<std::vector<std::pair<
        std::vector<float>,float>>>&);
};

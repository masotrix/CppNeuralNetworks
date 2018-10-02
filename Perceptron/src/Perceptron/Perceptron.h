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

    float forward(const std::vector<float> &x);

    void backward(float y, float y_pred);

    void step(float gamma);

    void train(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y, int EPOCHS,
        std::vector<float> &loss);

    float test(const std::vector<std::vector<float>> &X,
        const std::vector<float> &Y);

    static float testOp(
      const std::vector<float> &weights, float bias,
      const std::function<float(float)> &act,
      const std::function<float(float, float)> &op,
      const std::vector<std::vector<float>> &X);

};

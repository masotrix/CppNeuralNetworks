#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <random>
#include <utility>

class Dataset {

  protected:
    std::vector<std::vector<float>> _XTR,_YTR,_XTE,_YTE;

  public:
    std::pair<std::vector<std::vector<float>>,
     std::vector<std::vector<float>>> getTrainingExamples();
    std::pair<std::vector<std::vector<float>>,
     std::vector<std::vector<float>>> getTestingExamples();
};

class RandomDataset: public Dataset {
  public: RandomDataset(const int N, const int F, const int C);
};

#endif /* DATASET_H */

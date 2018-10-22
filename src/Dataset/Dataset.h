#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <random>
#include <utility>
#include <string>

class Dataset {

  protected:
    std::vector<std::pair<std::vector<float>,std::vector<float>>>
      _XTR,_XTE;

  public:
    std::vector<std::vector<std::pair<std::vector<float>,
     std::vector<float>>>> getTrainingExamples(
         int bsize=1, bool shuffle=true);
    const std::vector<std::pair<std::vector<float>,
     std::vector<float>>> getTestingExamples();
};

class RandomDataset: public Dataset {
  public: RandomDataset(const int N, const int F, const int C);
};

class UnseparatedDataset: public Dataset {
  protected:
    void separateData(
        const std::vector<std::pair<
          std::vector<float>,std::vector<float>>> &X,
        float testProportion);
  public:
    const std::vector<std::pair<std::vector<float>,
     std::vector<float>>> getTestingExamples();
};

class UnseparatedCSVDataset: public UnseparatedDataset {
  public: UnseparatedCSVDataset(std::string, float prop=0.1);
};

#endif /* DATASET_H */

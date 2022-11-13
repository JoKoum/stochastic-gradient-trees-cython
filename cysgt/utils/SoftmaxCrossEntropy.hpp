#ifndef SOFTMAX
#define SOFTMAX
#include <vector>
#include "GradHess.hpp"

class SoftmaxCrossEntropy
{   
    public:
    std::vector<GradHess> computeDerivatives(std::vector<double> groundTruth, std::vector<double> raw)
    {
        std::vector<GradHess> result(raw.size());
        std::vector<double> predictions = transfer(raw);
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = GradHess(predictions[i] - groundTruth[i], predictions[i] * (1.0 - predictions[i]));
        }
        return result;
    }
    
    std::vector<double> transfer(std::vector<double> raw)
    {
        std::vector<double> result(raw.size() + 1);
        for (int i = 0; i < raw.size(); i++)
        {
            result[i] = raw[i];
        }
        double max = -std::numeric_limits<double>::infinity();
        double sum = 0.0;
        for (int i = 0; i < result.size(); i++)
        {
            max = std::max(max,result[i]);
        }
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = exp(result[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.size(); i++)
        {
            result[i] /= sum;
        }
        return result;
    }
};

#endif
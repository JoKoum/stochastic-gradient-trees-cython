#ifndef SQUAREDERROR
#define SQUAREDERROR
#include <vector>
#include "GradHess.hpp"

class SquaredError
{
    public:
    std::vector<GradHess> computeDerivatives(std::vector<double> groundTruth, std::vector<double> raw)
    {
        std::vector<GradHess> result(raw.size());
        for (int i = 0; i < result.size(); i++)
        {
            result[i] = GradHess(raw[i] - groundTruth[i], 1.0);
        }
        return result;
    }
};

#endif
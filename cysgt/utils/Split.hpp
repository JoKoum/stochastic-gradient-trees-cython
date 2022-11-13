#ifndef SPLIT
#define SPLIT
#include <vector>
class Split
{
    // lossMean and lossVariance are actually statistics of the approximation to the *change* in loss.
    public:
    double lossMean = 0;
    double lossVariance = 0;
    std::vector<double> deltaPredictions;
    int feature = -1;
    int index = -1;

    void setLossMean(double lossMe)
    {
        lossMean = lossMe;
    }
    double getLossMean()
    {
        return lossMean;
    }

    void setLossVariance(double lossVar)
    {
        lossVariance = lossVar;
    }
    double getLossVariance()
    {
        return lossVariance;
    }
};

#endif
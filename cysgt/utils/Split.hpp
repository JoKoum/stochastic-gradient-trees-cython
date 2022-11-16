#ifndef SPLIT
#define SPLIT
#include <vector>
class Split
{
    // lossMean and lossVariance are actually statistics of the approximation to the *change* in loss.
    public:
    double lossMean;
    double lossVariance;
    std::vector<double> deltaPredictions;
    int feature;
    int index;

    Split()
    {
        lossMean = 0;
        lossVariance = 0;
        feature = -1;
        index = -1;
    }

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
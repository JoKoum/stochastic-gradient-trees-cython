#ifndef SGTNODE
#define SGTNODE
#include <vector>
#include <math.h>
#include "GradHess.hpp"
#include "GradHessStats.hpp"
#include "FeatureInfo.hpp"
#include "StreamingGradientTreeOptions.hpp"
#include "Split.hpp"

class Node
{
    public:
    Node *c;
    double mPrediction;
    std::vector<Node> mChildren;
    Split mSplit;
    GradHessStats mUpdateStats;
    std::vector<std::vector<GradHessStats>> mSplitStats;
    std::vector<FeatureInfo> mFeatureInfo;
    StreamingGradientTreeOptions mOptions;
    int mDepth;
    std::vector<bool> mHasSplit;
    int mInstances;
    bool isSplit;
    Node(double prediction, int depth, std::vector<bool> hasSplit, std::vector<FeatureInfo> featureInfo, StreamingGradientTreeOptions options);
    ~Node();
    void reset();
    Node* getLeaf(std::vector<int> features);
    void update(std::vector<int> features, GradHess gradHess);
    double predict();
    bool checkIfSplit();
    Split findBestSplit();
    void applySplit(Split split);
    protected:
    double computeDeltaPrediction(GradHess gradHess);
};

#endif
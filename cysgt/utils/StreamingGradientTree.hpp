#ifndef SGT
#define SGT
#include <vector>
#include <math.h>
#include "GradHess.hpp"
#include "GradHessStats.hpp"
#include "FeatureInfo.hpp"
#include "StreamingGradientTreeOptions.hpp"
#include "Statistics.hpp"
#include "Split.hpp"
#include "Node.hpp"

class StreamingGradientTree
{   
    protected :
    double computePValue(Split split, int instances);
    public :
    std::vector<FeatureInfo> mFeatureInfo;
    StreamingGradientTreeOptions mOptions;
    Node *mRoot;
    int mNumNodes;
    int mNumNodeUpdates;
    int mMaxDepth;
    int mNumSplits;
    StreamingGradientTree(std::vector<FeatureInfo> featureInfo, StreamingGradientTreeOptions options);
    ~StreamingGradientTree();
    void deleteNodes(Node *node);
    int getNumNodes();
    int getNumNodeUpdates();
    int getNumSplits();
    int getDepth();
    void update(std::vector<int> features, GradHess gradHess);
    double predict(std::vector<int> features);
};

#endif
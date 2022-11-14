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

class StreamingGradientTree
{   
    protected :
    double computePValue(Split split, int instances);
    public :
    std::vector<FeatureInfo> mFeatureInfo;
    StreamingGradientTreeOptions mOptions;
    class StreamingGradientTree::Node;
    StreamingGradientTree::Node* mRoot;
    int mNumNodes;
    int mNumNodeUpdates;
    int mMaxDepth;
    int mNumSplits;
    StreamingGradientTree(std::vector<FeatureInfo> featureInfo, StreamingGradientTreeOptions options);
    int getNumNodes();
    int getNumNodeUpdates();
    int getNumSplits();
    int getDepth();
    void update(std::vector<int> features, GradHess gradHess);
    double predict(std::vector<int> features);
    class Node
    {
        private :
        StreamingGradientTree *parentClass;
        protected:
        double mPrediction;
        std::vector<Node*> mChildren;
        Split mSplit;
        GradHessStats mUpdateStats;
        std::vector<std::vector<GradHessStats>> mSplitStats;
        int mDepth;
        std::vector<bool> mHasSplit;
        public :
        int mInstances;
        Node(double prediction, int depth, std::vector<bool> hasSplit, StreamingGradientTree *parentTree);
        void reset();
        Node* getLeaf(std::vector<int> features);
        void update(std::vector<int> features, GradHess gradHess);
        double predict();
        Split findBestSplit();
        void applySplit(Split split);
        protected:
        double computeDeltaPrediction(GradHess gradHess);
    };
};

#endif
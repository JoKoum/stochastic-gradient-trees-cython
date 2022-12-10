#include <vector>
#include <iostream>
#include <math.h>
#include "GradHess.hpp"
#include "GradHessStats.hpp"
#include "FeatureInfo.hpp"
#include "StreamingGradientTreeOptions.hpp"
#include "Statistics.cpp"
#include "Split.hpp"
#include "StreamingGradientTree.hpp"
#include "Node.cpp"

StreamingGradientTree::StreamingGradientTree(std::vector<FeatureInfo> featureInfo, StreamingGradientTreeOptions options)
{
    mFeatureInfo = featureInfo;
    mOptions = options;
    std::vector<bool> hasSplit(mFeatureInfo.size());
    for (int i = 0; i < hasSplit.size(); i++)
    {
        hasSplit[i] = false;
    }
    mNumNodes = 0;
    mMaxDepth = 0;
    mRoot = new Node(options.initialPrediction, 1, hasSplit, mFeatureInfo, mOptions);

}

StreamingGradientTree::~StreamingGradientTree()
{
    deleteNodes(mRoot);
}

void StreamingGradientTree::deleteNodes(Node *node)
{
    if (node->mChildren[0] != NULL)
    {
        deleteNodes(node->mChildren[0]);
    }

    if (node->mChildren[1] != NULL)
    {
        deleteNodes(node->mChildren[1]);
    }

    delete node;
}

int StreamingGradientTree::getNumNodes()
{
    return mNumNodes;
}

int StreamingGradientTree::getDepth()
{
    return mMaxDepth;
}

void StreamingGradientTree::update(std::vector<int> features, GradHess gradHess)
{
    Node *leaf = mRoot->getLeaf(features);
    leaf->update(features, gradHess);
    
    if (leaf->mInstances % mOptions.gracePeriod != 0)
    {
        return;
    }
    Split bestSplit = leaf->findBestSplit();

    double p = computePValue(bestSplit, leaf->mInstances);
    
    if (p < mOptions.delta && bestSplit.lossMean < 0.0)
    {
        leaf->applySplit(bestSplit);
        if (mFeatureInfo[0].type == FeatureType::nominal)
        {
            mNumNodes += leaf->mChildren.size();
        }
        else
        {
            mNumNodes += 2;
        }
        mMaxDepth++ ;
    }
    leaf = nullptr;
}

double StreamingGradientTree::predict(std::vector<int> features)
{
    return mRoot->getLeaf(features)->predict();
}

double StreamingGradientTree::computePValue(Split split, int instances)
{
    // H0: the expected loss is zero
    // HA: the expected loss is not zero
    try
    {
        double F = instances * pow(split.lossMean, 2.0) / split.lossVariance;
        return Statistics::FProbability(F, 1, instances - 1);
    }
    catch (std::logic_error e)
    {
        // System.err.println(e.getMessage());
        // System.err.println(split.lossMean + " " + split.lossVariance);
        return 1.0;
    }
}

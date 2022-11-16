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
    mRoot = new StreamingGradientTree::Node(options.initialPrediction, 1, hasSplit, this);

}

StreamingGradientTree::~StreamingGradientTree()
{
    delete mRoot;
}

int StreamingGradientTree::getNumNodes()
{
    return mNumNodes;
}

int StreamingGradientTree::getNumNodeUpdates()
{
    return mNumNodeUpdates;
}

int StreamingGradientTree::getNumSplits()
{
    return mNumSplits;
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

StreamingGradientTree::Node::Node(double prediction, int depth, std::vector<bool> hasSplit, StreamingGradientTree *parentTree)
{
    mPrediction = prediction;
    this->parentClass = parentTree;
    parentClass->mNumNodes++;
    mDepth = depth;
    parentClass->mMaxDepth = std::max(parentClass->mMaxDepth, mDepth);
    mHasSplit = hasSplit;
    reset();
}

StreamingGradientTree::Node::~Node()
{
    delete parentClass;
}

void StreamingGradientTree::Node::reset()
{
    std::vector<std::vector<GradHessStats>> splitStats(parentClass->mFeatureInfo.size());
    mUpdateStats =  GradHessStats();
    mInstances = 0;

    for (int i = 0; i < splitStats.size(); i++)
    {
        splitStats[i] = std::vector<GradHessStats>(parentClass->mFeatureInfo[i].categories);
        
        for (int j = 0; j < splitStats[i].size(); j++)
        {
            splitStats[i][j] = GradHessStats();
        }
    }
    mSplitStats = splitStats;
    std::vector<std::vector<GradHessStats>>().swap(splitStats);
}

StreamingGradientTree::Node* StreamingGradientTree::Node::getLeaf(std::vector<int> features)
{
    if (mChildren.empty())
    {
        return this;
    }
    else 
    {
        FeatureType featureType = parentClass->mFeatureInfo[mSplit.feature].type;
        Node *c = nullptr;

        if (features[mSplit.feature] == -1)
        {
            c = mChildren[0];
        }
        else if (featureType == FeatureType::nominal)
        {
            c = mChildren[features[mSplit.feature]];
        }
        else if (featureType == FeatureType::ordinal)
        {
            if (features[mSplit.feature] <= mSplit.index)
            {
                c = mChildren[0];
            }
            else 
            {
                c = mChildren[1];
            }
        }
        else 
        {
            std::cout << "Unhandled attribute type" << std::endl;
        }
        return c->getLeaf(features);
    }
}

void StreamingGradientTree::Node::update(std::vector<int> features, GradHess gradHess)
{
    mInstances++;
    for (int i = 0; i < features.size(); i++)
    {
        if (features[i] == -1)
        {
            continue;
        }
        mSplitStats[i][features[i]].addObservation(gradHess);
    }
    mUpdateStats.addObservation(gradHess);
}

double StreamingGradientTree::Node::predict()
{
    return mPrediction;
}

Split StreamingGradientTree::Node::findBestSplit()
{
    Split best = Split();
    // We can try to update the prediction using the new gradient information
    best.deltaPredictions = std::vector<double>{computeDeltaPrediction(mUpdateStats.getMean())};
    best.lossMean = mUpdateStats.getDeltaLossMean(best.deltaPredictions[0]);
    best.lossVariance = mUpdateStats.getDeltaLossVariance(best.deltaPredictions[0]);
    best.feature = -1;
    best.index = -1;
    for (int i = 0; i < mSplitStats.size(); i++)
    {
        Split candidate = Split();
        candidate.feature = i;
        
        if (parentClass->mFeatureInfo[i].type == FeatureType::nominal)
        {
            if (mHasSplit[i])
            {
                continue;
            }
            candidate.deltaPredictions = std::vector<double>(mSplitStats[i].size());
            double lossMean = 0.0;
            double lossVar = 0.0;
            int observations = 0;
            
            for (int j = 0; j < mSplitStats[i].size(); j++)
            {
                double p = computeDeltaPrediction(mSplitStats[i][j].getMean());
                double m = mSplitStats[i][j].getDeltaLossMean(p);
                double s = mSplitStats[i][j].getDeltaLossVariance(p);
                int n = mSplitStats[i][j].getObservationCount();
                candidate.deltaPredictions[j] = p;
                lossMean = GradHessStats::combineMean(lossMean, observations, m, n);
                lossVar = GradHessStats::combineVariance(lossMean, lossVar, observations, m, s, n);
                observations += n;
            }
            candidate.lossMean = lossMean + mSplitStats[i].size() * parentClass->mOptions.gamma / mInstances;
            candidate.lossVariance = lossVar;
        }
        else if (parentClass->mFeatureInfo[i].type == FeatureType::ordinal)
        {
            std::vector<GradHessStats> forwardCumulativeSum(parentClass->mFeatureInfo[i].categories - 1);
            std::vector<GradHessStats> backwardCumulativeSum(parentClass->mFeatureInfo[i].categories - 1);
            // Compute the split stats for each possible split point
            for (int j = 0; j < parentClass->mFeatureInfo[i].categories - 1; j++)
            {
                forwardCumulativeSum[j] = GradHessStats();
                forwardCumulativeSum[j].add(mSplitStats[i][j]);
                if (j > 0)
                {
                    forwardCumulativeSum[j].add(forwardCumulativeSum[j - 1]);
                }
            }
            for (int j = parentClass->mFeatureInfo[i].categories - 2; j >= 0; j--)
            {
                backwardCumulativeSum[j] = GradHessStats();
                backwardCumulativeSum[j].add(mSplitStats[i][j + 1]);
                if (j + 1 < backwardCumulativeSum.size())
                {
                    backwardCumulativeSum[j].add(backwardCumulativeSum[j + 1]);
                }
            }
            candidate.lossMean = std::numeric_limits<double>::infinity();
            candidate.deltaPredictions = std::vector<double>(2);
            for (int j = 0; j < forwardCumulativeSum.size(); j++)
            {               
                double deltaPredLeft = computeDeltaPrediction(forwardCumulativeSum[j].getMean());
                double lossMeanLeft = forwardCumulativeSum[j].getDeltaLossMean(deltaPredLeft);
                double lossVarLeft = forwardCumulativeSum[j].getDeltaLossVariance(deltaPredLeft);
                int numLeft = forwardCumulativeSum[j].getObservationCount();
                double deltaPredRight = computeDeltaPrediction(backwardCumulativeSum[j].getMean());
                double lossMeanRight = backwardCumulativeSum[j].getDeltaLossMean(deltaPredRight);
                double lossVarRight = backwardCumulativeSum[j].getDeltaLossVariance(deltaPredRight);
                int numRight = backwardCumulativeSum[j].getObservationCount();
                double lossMean = GradHessStats::combineMean(lossMeanLeft, numLeft, lossMeanRight, numRight);
                double lossVar = GradHessStats::combineVariance(lossMeanLeft, lossVarLeft, numLeft, lossMeanRight, lossVarRight, numRight);
                
                if (lossMean < candidate.lossMean)
                {
                    candidate.lossMean = lossMean + 2.0 * parentClass->mOptions.gamma / mInstances;
                    candidate.lossVariance = lossVar;
                    candidate.index = j;
                    candidate.deltaPredictions[0] = deltaPredLeft;
                    candidate.deltaPredictions[1] = deltaPredRight;
                }
            }
        }
        else 
        {
            std::cout << "Unhandled attribute type" << std::endl;
        }
        if (candidate.lossMean < best.lossMean)
        {
            best = candidate;
        }
    }
    return best;
}

void StreamingGradientTree::Node::applySplit(Split split)
{
    // Should we just update the prediction being made?
    if (split.feature == -1)
    {
        mPrediction += split.deltaPredictions[0];
        parentClass->mNumNodeUpdates++;
        reset();
        return;
    }

    mSplit = split;
    parentClass->mNumSplits++;
    mHasSplit[split.feature] = true;

    if (parentClass->mFeatureInfo[split.feature].type == nominal)
    {
        std::vector<Node*> children(parentClass->mFeatureInfo[split.feature].categories);
        mChildren = children;
        for (int i = 0; i < mChildren.size(); i++)
        {
            mChildren[i] = new Node(mPrediction + split.deltaPredictions[i], mDepth + 1, mHasSplit, parentClass);
        }
        std::vector<Node*>().swap(children);
    }
    else if (parentClass->mFeatureInfo[split.feature].type == FeatureType::ordinal)
    {
        std::vector<Node*> children(2);
        mChildren = children;
        mChildren[0] =  new Node(mPrediction + split.deltaPredictions[0], mDepth + 1, mHasSplit, parentClass);
        mChildren[1] =  new Node(mPrediction + split.deltaPredictions[1], mDepth + 1, mHasSplit, parentClass);
        std::vector<Node*>().swap(children);
    }
    else 
    {
        std::cout << "Unhandled attribute type" << std::endl;
    }
    // Free up memory used by the split stats
    mSplitStats.clear();
}

double StreamingGradientTree::Node::computeDeltaPrediction(GradHess gradHess)
{
    return -gradHess.gradient / (gradHess.hessian + 2.2250738585072014E-308 + parentClass->mOptions.lambda);
}
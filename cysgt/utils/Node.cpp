#include <vector>
#include <iostream>
#include <math.h>
#include "GradHess.hpp"
#include "GradHessStats.hpp"
#include "FeatureInfo.hpp"
#include "StreamingGradientTreeOptions.hpp"
#include "Split.hpp"
#include "Node.hpp"


Node::Node(double prediction, int depth, std::vector<bool> hasSplit, std::vector<FeatureInfo> featureInfo, StreamingGradientTreeOptions options)
{
    mPrediction = prediction;
    mDepth = depth;
    mHasSplit = hasSplit;
    mFeatureInfo = featureInfo;
    mOptions = options;
    reset();
}

Node::~Node()
{
    mChildren.clear();
}

void Node::reset()
{
    std::vector<std::vector<GradHessStats>> splitStats(mFeatureInfo.size());
    mUpdateStats =  GradHessStats();
    mInstances = 0;

    for (int i = 0; i < splitStats.size(); i++)
    {
        splitStats[i] = std::vector<GradHessStats>(mFeatureInfo[i].categories);
        
        for (int j = 0; j < splitStats[i].size(); j++)
        {
            splitStats[i][j] = GradHessStats();
        }
    }
    mSplitStats = splitStats;
    std::vector<std::vector<GradHessStats>>().swap(splitStats);
}

Node* Node::getLeaf(std::vector<int> features)
{
    if (mChildren.empty())
    {
        return this;
    }
    else 
    {
        FeatureType featureType = mFeatureInfo[mSplit.feature].type;
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

void Node::update(std::vector<int> features, GradHess gradHess)
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

double Node::predict()
{
    return mPrediction;
}

bool Node::checkIfSplit()
{
    return isSplit;
}

Split Node::findBestSplit()
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
        
        if (mFeatureInfo[i].type == FeatureType::nominal)
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
            candidate.lossMean = lossMean + mSplitStats[i].size() * mOptions.gamma / mInstances;
            candidate.lossVariance = lossVar;
        }
        else if (mFeatureInfo[i].type == FeatureType::ordinal)
        {
            std::vector<GradHessStats> forwardCumulativeSum(mFeatureInfo[i].categories - 1);
            std::vector<GradHessStats> backwardCumulativeSum(mFeatureInfo[i].categories - 1);
            // Compute the split stats for each possible split point
            for (int j = 0; j < mFeatureInfo[i].categories - 1; j++)
            {
                forwardCumulativeSum[j] = GradHessStats();
                forwardCumulativeSum[j].add(mSplitStats[i][j]);
                if (j > 0)
                {
                    forwardCumulativeSum[j].add(forwardCumulativeSum[j - 1]);
                }
            }
            for (int j = mFeatureInfo[i].categories - 2; j >= 0; j--)
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
                    candidate.lossMean = lossMean + 2.0 * mOptions.gamma / mInstances;
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

void Node::applySplit(Split split)
{
    // Should we just update the prediction being made?
    if (split.feature == -1)
    {
        mPrediction += split.deltaPredictions[0];
        isSplit = false;
        reset();
        return;
    }

    mSplit = split;
    mHasSplit[split.feature] = true;

    if (mFeatureInfo[split.feature].type == FeatureType::nominal)
    {
        for (int i = 0; i < mChildren.size(); i++)
        {
            mChildren.push_back(new Node(mPrediction + split.deltaPredictions[i], mDepth + 1, mHasSplit, mFeatureInfo, mOptions));
        }
    }
    else if (mFeatureInfo[split.feature].type == FeatureType::ordinal)
    {
        mChildren.push_back(new Node(mPrediction + split.deltaPredictions[0], mDepth + 1, mHasSplit, mFeatureInfo, mOptions));
        mChildren.push_back(new Node(mPrediction + split.deltaPredictions[1], mDepth + 1, mHasSplit, mFeatureInfo, mOptions));
    }
    else 
    {
        std::cout << "Unhandled attribute type" << std::endl;
    }
    isSplit = true;
    // Free up memory used by the split stats
    mSplitStats.clear();
}

double Node::computeDeltaPrediction(GradHess gradHess)
{
    return -gradHess.gradient / (gradHess.hessian + 2.2250738585072014E-308 + mOptions.lambda);
}
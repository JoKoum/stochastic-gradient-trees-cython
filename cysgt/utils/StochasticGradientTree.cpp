#include "StreamingGradientTree.cpp"
#include "StreamingGradientTreeOptions.hpp"
#include "FeatureInfo.hpp"
#include "FeatureType.hpp"
#include "SquaredError.hpp"
#include "SoftmaxCrossEntropy.hpp"
#include "StochasticGradientTree.hpp"
#include <iostream>
#include <algorithm>

StochasticGradientTree::StochasticGradientTree(std::string ob, int binNo = 64, int batch_size = 200, int epochNo = 20, double l = 0.1, double g = 1.0, std::vector<double> upper = {}, std::vector<double> lower = {}, double lr = 1)
{
    bins = binNo;
    batchSize = batch_size;
    epochs = epochNo;
    lambda = l;
    gamma = g;
    lower_bounds = lower;
    upper_bounds = upper;
    learning_rate = l;
    obType = ob;

    if (obType == "classification")
    {
       softmaxObjective = new SoftmaxCrossEntropy();
    }
    else if (obType == "regression")
    {
        squaredObjective = new SquaredError();
    }

    options = new StreamingGradientTreeOptions();
    options->gracePeriod = batch_size;
    options->lambda = lambda;
    options->gamma = gamma;

    if ((upper_bounds.size() == 0) & (lower_bounds.size() == 0))
    {
        MinMaxProvided = false;
    }
    else
    {
        MinMaxProvided = true;
    }

    isFit = false;

    samplesSeen = 0;

}

int StochasticGradientTree::getEpochs()
{
    return epochs;
}

void StochasticGradientTree::setEpochs(int ep)
{
    epochs = ep;
}
        
int StochasticGradientTree::getBins()
{
    return bins;
}

void StochasticGradientTree::setBins(int b)
{
    bins = b;
}

void StochasticGradientTree::setFit(bool fit)
{
    isFit = fit;
}

bool StochasticGradientTree::getFit()
{
    return isFit;
}

void StochasticGradientTree::setTrainBatchSize(int bs)
{
    batchSize = bs;
}

int StochasticGradientTree::getTrainBatchSize()
{
    return batchSize;
}

void StochasticGradientTree::setLambda(double l)
{
    lambda = l;
}
        
double StochasticGradientTree::getLambda()
{
    return lambda;
}

void StochasticGradientTree::setGamma(double g)
{
    gamma = g;
}

double StochasticGradientTree::getGamma()
{
    return gamma;
}

int StochasticGradientTree::getDepth()
{
    return tree->getDepth();
}

int StochasticGradientTree::getTotalNodes()
{
    return tree->getNumNodes();
}

void StochasticGradientTree::setLearningRate(double lr)
{
    learning_rate = lr;
}

double StochasticGradientTree::getLearningRate()
{
    return learning_rate;
}

void StochasticGradientTree::setBounds(std::vector<double> u, std::vector<double> l)
{
    upper_bounds = u;
    lower_bounds = l;
}

int StochasticGradientTree::getIsFit()
{
    return isFit;
}

int StochasticGradientTree::getClassifierType()
{
    int result;

    if (obType == "classification")
    {
        result = 0;
    }
    else if (obType == "regression")
    {
        result = 1;
    }
    
    return result;
}

std::vector<std::vector<int>> StochasticGradientTree::createFeatures(std::vector<std::vector<double>> X, std::vector<double> u, std::vector<double> l)
{
    
    if (samplesSeen < 1000)
    {
        if (!isFit)
        {
            features = X;
        }
        else
        {
            features.insert(features.end(), X.begin(), X.end());
        }
    }

    samplesSeen += X.size();
    
    if (!MinMaxProvided)
    {
        upper_bounds = u;
        lower_bounds = l;
    }

    if (!isFit)
    {
        FeatureType ordinal = FeatureType::ordinal;
        FeatureInfo* fInfo = new FeatureInfo();
        fInfo->setFeatureType(ordinal);
        fInfo->setCategories(bins);

        for (int i = 0; i < X[0].size(); i++)
        {
            featureInfo.push_back(fInfo);
            buckets.push_back(bins);
        }
            
    }

    std::vector<std::vector<int>> discretized(X.size());

    for (int i = 0; i < X.size(); i++)
    {
        discretized[i] = discretize(X[i]);
    }

    return discretized;

}
std::vector<int> StochasticGradientTree::discretize(std::vector<double> observations)
{
    std::vector<int> discretized = {};
    double scaling;

    for (int i = 0; i < observations.size(); i++)
    {
        if (upper_bounds[i] == lower_bounds[i])
        {
            scaling = 1;
        }
        else
        {
            scaling = (observations[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i]);
        }

        scaled_observations = (int)(buckets[i] * scaling);
        scaled_observations = std::min(buckets[i] -1, std::max(0, scaled_observations));
        discretized.push_back(scaled_observations);
    }

    return discretized;
}

void StochasticGradientTree::train(std::vector<int> x, double y)
{  
    double pred = tree->predict(x);

    std::vector<double> prediction = {pred};
    std::vector<double> groundTruth = {y};

    std::vector<GradHess> gradHess;
    
    if (obType == "classification")
    {
        gradHess = softmaxObjective->computeDerivatives(groundTruth, prediction);
    }
    else if (obType == "regression")
    {
        gradHess = squaredObjective->computeDerivatives(groundTruth, prediction);
    }
    
    GradHess* gh = &gradHess[0];
    tree->update(x, gh);

}

void StochasticGradientTree::fit(std::vector<std::vector<double>> X, std::vector<double> y)
{
    std::vector<std::vector<int>> features(X.size());
    features = createFeatures(X, upper_bounds, lower_bounds);
    
    if (!isFit)
    {
        tree = new StreamingGradientTree(featureInfo, options);
    }

    for (int e = 1; e < epochs + 1; e++)
    {    
        for (int i = 0; i < features.size(); i++)
        {
            train(features[i], y[i]);
        }
    }
    isFit = true;
}

std::vector<std::vector<double>> StochasticGradientTree::predictProba(std::vector<std::vector<double>> X)
{
    std::vector<std::vector<int>> features(X.size());
    features = createFeatures(X, upper_bounds, lower_bounds);

    std::vector<double> logits(X.size()); 
    std::vector<std::vector<double>> probs(X.size()), proba(X.size());

    if (!isFit)
    {
        tree = new StreamingGradientTree(featureInfo, options);
    }
    
    for (int i = 0; i < features.size(); i++)
    {
        logits[i] = tree->predict(features[i]);
    }

    if (obType == "classification")
    {
        for (int i = 0; i < logits.size(); i++)
        {
            probs[i] = softmaxObjective->transfer({logits[i]});
        }

        for (int i = 0; i < probs.size(); i++)
        {
            proba[i] = {probs[i][1], probs[i][0]};
        }
    }
    else if (obType == "regression")
    {
        for (int i = 0; i < logits.size(); i++)
        {
            proba[i] = {logits[i], logits[i]};
        }
    }

    return proba; 

}
#ifndef STOCHGT
#define STOCHGT

#include <vector>
#include "StreamingGradientTree.hpp"
#include "StreamingGradientTreeOptions.hpp"
#include "FeatureInfo.hpp"
#include "SoftmaxCrossEntropy.hpp"
#include "SquaredError.hpp"
#include <string>

class StochasticGradientTree
{
    public:
        StreamingGradientTreeOptions options;
        StreamingGradientTree *tree;
        SoftmaxCrossEntropy softmaxObjective;
        SquaredError squaredObjective;
        std::vector<FeatureInfo> featureInfo;
        FeatureInfo fInfo;
        std::string obType;
        int scaled_observations;
        std::vector<int> buckets;
        int bins;
        int batchSize;
        int epochs;
        double mLambda;
        double gamma;
        std::vector<double> upper_bounds;
        std::vector<double> lower_bounds;
        double learning_rate;
        bool MinMaxProvided;
        std::vector<std::vector<double>> samples;

        StochasticGradientTree(std::string ob, int binNo, int batch_size, int epochNo, double l, double g, std::vector<double> upper, std::vector<double> lower, double lr);
        int getEpochs();
        void setEpochs(int ep);
        int getBins();
        void setBins(int b);
        void setFit(bool fit);
        bool getFit();
        void setTrainBatchSize(int bs);
        int getTrainBatchSize();
        void setLambda(double l);
        double getLambda();
        void setGamma(double g);
        double getGamma();
        int getDepth();
        int getTotalNodes();
        void setLearningRate(double lr);
        double getLearningRate();
        void setBounds(std::vector<double> u, std::vector<double> l);
        int getIsFit();
        int getClassifierType();
        std::vector<std::vector<int>> createFeatures(std::vector<std::vector<double>> X, std::vector<double> u, std::vector<double> l);
        std::vector<int> discretize(std::vector<double> observations);
        void train(std::vector<int> x, double y);
        void fit(std::vector<std::vector<double>> X, std::vector<double> y);
        std::vector<std::vector<double>> predictProba(std::vector<std::vector<double>> X);
    private:
        bool isFit;
        size_t samplesSeen;

};
#endif
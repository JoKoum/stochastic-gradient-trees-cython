#ifndef SGTOPTIONS
#define SGTOPTIONS
class StreamingGradientTreeOptions
{
    public:
    double delta = 1E-7;
    int gracePeriod = 200;
    double initialPrediction = 0.0;
    double lambda = 0.1;
    double gamma = 1.0;

    void setDelta(double d)
    {
        delta = d;
    }

    double getDelta()
    {
        return delta;
    }
   
    void setGracePeriod(int period)
    {
        gracePeriod = period;
    }
    
    int getGracePeriod()
    {
        return gracePeriod;
    }
    
    void setInitialPrediction(double initPred)
    {
        initialPrediction = initPred;
    }

    double getInitialPrediction()
    {
        return initialPrediction;
    }

    void setLambda(double l)
    {
        lambda = l;
    }

    double getLambda()
    {
        return lambda;
    }

    void setGamma(double g)
    {
        gamma = g;
    }

    double getGamma()
    {
        return gamma;
    }
};
#endif
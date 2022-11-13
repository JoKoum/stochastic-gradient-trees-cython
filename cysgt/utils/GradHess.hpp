#ifndef GRADHESS
#define GRADHESS

class GradHess
{
    public:
    double gradient;
    double hessian;
    GradHess()
    {
        gradient = 0;
        hessian = 0;
    }    
    GradHess(GradHess *gradHess)
    {
        gradient = gradHess->gradient;
        hessian = gradHess->hessian;
    }   
    GradHess(double grad, double hess)
    {
        gradient = grad;
        hessian = hess;
    }
    void add(GradHess *gradHess)
    {
        gradient += gradHess->gradient;
        hessian += gradHess->hessian;
    }
    void sub(GradHess *gradHess)
    {
        gradient -= gradHess->gradient;
        hessian -= gradHess->hessian;
    }
};
#endif
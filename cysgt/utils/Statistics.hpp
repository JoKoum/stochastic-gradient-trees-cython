#ifndef STATISTICS
#define STATISTICS
#define _USE_MATH_DEFINES
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <math.h>

class Statistics
{
    // * Some constants 
    protected:
    static double MACHEP;
    static double MAXLOG;
    static double MINLOG;
    static double MAXGAM;
    static double SQTPI;
    static double SQRTH;
    static double LOGPI;
    static double big;
    static double biginv;
    // ************************************************
    //     * COEFFICIENTS FOR METHOD normalInverse() *
    //     ************************************************
    // approximation for 0 <= |y - 0.5| <= 3/8
    static std::vector<double> P0;
    static std::vector<double> Q0;
    //     * Approximation for interval z = sqrt(-2 log y ) between 2 and 8 i.e., y
    //     * between exp(-2) = .135 and exp(-32) = 1.27e-14.
    static std::vector<double> P1;
    static std::vector<double> Q1;
    //     * Approximation for interval z = sqrt(-2 log y ) between 8 and 64 i.e., y
    //     * between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    static std::vector<double> P2;
    static std::vector<double> Q2;
    
    public:
    static double binomialStandardError(double p, int n)  
    // *
    //     * Computes standard error for observed values of a binomial random variable.
    //     * 
    //     * @param p the probability of success
    //     * @param n the size of the sample
    //     * @return the standard error
    {
        if (n == 0)
        {
            return 0;
        }
        return sqrt((p * (1 - p)) / n);
    }
    static double chiSquaredProbability(double x, double v)
    // *
    //     * Returns chi-squared probability for given value and degrees of freedom.
    //     * (The probability that the chi-squared variate will be greater than x for
    //     * the given degrees of freedom.)
    //     * 
    //     * @param x the value
    //     * @param v the number of degrees of freedom
    //     * @return the chi-squared probability
    {
        if (x < 0.0 || v < 1.0)
        {
            return 0.0;
        }
        return incompleteGammaComplement(v / 2.0, x / 2.0);
    }
    static double FProbability(double F, int df1, int df2)
    // *
    //     * Computes probability of F-ratio.
    //     * 
    //     * @param F the F-ratio
    //     * @param df1 the first number of degrees of freedom
    //     * @param df2 the second number of degrees of freedom
    //     * @return the probability of the F-ratio.
    
    {
        return incompleteBeta(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * F));
    }
    static double normalProbability(double a)
    // *
    //     * Returns the area under the Normal (Gaussian) probability density function,
    //     * integrated from minus infinity to <tt>x</tt> (assumes mean is zero,
    //     * variance is one).
    //     * 
    //     * <pre>
    //     *                            x
    //     *                             -
    //     *                   1        | |          2
    //     *  normal(x)  = ---------    |    exp( - t /2 ) dt
    //     *               sqrt(2pi)  | |
    //     *                           -
    //     *                          -inf.
    //     * 
    //     *             =  ( 1 + erf(z) ) / 2
    //     *             =  erfc(z) / 2
    //     * </pre>
    //     * 
    //     * where <tt>z = x/sqrt(2)</tt>. Computation is via the functions
    //     * <tt>errorFunction</tt> and <tt>errorFunctionComplement</tt>.
    //     * 
    //     * @param a the z-value
    //     * @return the probability of the z value according to the normal pdf
    {
        double x;
        double y;
        double z;

        x = a * SQRTH;
        z = abs(x);

        if (z < SQRTH)
        {
            y = 0.5 + 0.5 * errorFunction(x);
        }
        else 
        {
            y = 0.5 * errorFunctionComplemented(z);
            if (x > 0)
            {
                y = 1.0 - y;
            }
        }
        return y;
    }

    static double normalInverse(double y0)
    // *
    //     * Returns the value, <tt>x</tt>, for which the area under the Normal
    //     * (Gaussian) probability density function (integrated from minus infinity to
    //     * <tt>x</tt>) is equal to the argument <tt>y</tt> (assumes mean is zero,
    //     * variance is one).
    //     * <p>
    //     * For small arguments <tt>0 < y < exp(-2)</tt>, the program computes
    //     * <tt>z = sqrt( -2.0 * log(y) )</tt>; then the approximation is
    //     * <tt>x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z)</tt>. There are two rational
    //     * functions P/Q, one for <tt>0 < y < exp(-32)</tt> and the other for
    //     * <tt>y</tt> up to <tt>exp(-2)</tt>. For larger arguments,
    //     * <tt>w = y - 0.5</tt>, and <tt>x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2))</tt>.
    //     * 
    //     * @param y0 the area under the normal pdf
    //     * @return the z-value
    {
        double x;
        double y;
        double z;
        double y2;
        double x0;
        double x1;

        int code;

        double s2pi = sqrt(2.0 * M_PI);

        if (y0 <= 0.0)
        {
            throw std::logic_error("IllegalArgumentException");
        }
        if (y0 >= 1.0)
        {
            throw std::logic_error("IllegalArgumentException");
        }
        code = 1;
        y = y0;
        if (y > (1.0 - 0.13533528323661269189))
        {
            // 0.135... = exp(-2) 
            y = 1.0 - y;
            code = 0;
        }

        if (y > 0.13533528323661269189)
        {
            y = y - 0.5;
            y2 = y * y;
            x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8));
            x = x * s2pi;
            return (x);
        }

        x = sqrt(-2.0 * log(y));
        x0 = x - log(x) / x;

        z = 1.0 / x;
        if (x < 8.0)
        {
            x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8);
        }
        else 
        {
            x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8);
        }
        x = x0 - x1;
        if (code != 0)
        {
            x = -x;
        }
        return (x);
    }

    static double lnGamma(double x)
    // *
    //     * Returns natural logarithm of gamma function.
    //     * 
    //     * @param x the value
    //     * @return natural logarithm of gamma function
    {
        double p;
        double q;
        double w;
        double z;

        std::vector<double> A{8.11614167470508450300E-4, -5.95061904284301438324E-4,
        7.93650340457716943945E-4, -2.77777777730099687205E-3,
        8.33333333333331927722E-2};
        std::vector<double> B{-1.37825152569120859100E3, -3.88016315134637840924E4,
        -3.31612992738871184744E5, -1.16237097492762307383E6,
        -1.72173700820839662146E6, -8.53555664245765465627E5};
        std::vector<double> C{-3.51815701436523470549E2, -1.70642106651881159223E4,
        -2.20528590553854454839E5, -1.13933444367982507207E6,
        -2.53252307177582951285E6, -2.01889141433532773231E6};
        
        if (x < -34.0)
        {
            q = -x;
            w = lnGamma(q);
            p = floor(q);
            if (p == q)
            {
                throw std::logic_error("lnGamma: Overflow");
            }
            z = q - p;
            if (z > 0.5)
            {
                p += 1.0;
                z = p - q;
            }
            z = q * sin(M_PI * z);
            if (z == 0.0)
            {
                throw std::logic_error("lnGamma: Overflow");
            }
            z = LOGPI - log(z) - w;
            return z;
        }

        if (x < 13.0)
        {
            z = 1.0;
            while (x >= 3.0)
            {
                x -= 1.0;
                z *= x;
            }
            while (x < 2.0)
            {
                if (x == 0.0)
                {
                    throw std::logic_error("lnGamma: Overflow");
                }
                z /= x;
                x += 1.0;
            }
            if (z < 0.0)
            {
                z = -z;
            }
            if (x == 2.0)
            {
                return log(z);
            }
            x -= 2.0;
            p = x * polevl(x, B, 5) / p1evl(x, C, 6);
            return (log(z) + p);
        }

        if (x > 2.556348E305)
        {
            throw std::logic_error("lnGamma: Overflow");
        }

        q = (x - 0.5) * log(x) - x + 0.91893853320467274178;
        
        if (x > 1.0E8)
        {
            return (q);
        }
        
        p = 1.0 / (x * x);
        if (x >= 1000.0)
        {
            q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p + 0.0833333333333333333333) / x;
        }
        else 
        {
            q += polevl(p, A, 4) / x;
        }
        return q;
    }

    static double errorFunction(double x)
    // *
    //     * Returns the error function of the normal distribution. The integral is
    //     * 
    //     * <pre>
    //     *                           x 
    //     *                            -
    //     *                 2         | |          2
    //     *   erf(x)  =  --------     |    exp( - t  ) dt.
    //     *              sqrt(pi)   | |
    //     *                          -
    //     *                           0
    //     * </pre>
    //     * 
    //     * <b>Implementation:</b> For
    //     * <tt>0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2)</tt>; otherwise
    //     * <tt>erf(x) = 1 - erfc(x)</tt>.
    //     * <p>
    //     * Code adapted from the <A
    //     * HREF="http://www.sci.usq.edu.au/staff/leighb/graph/Top.html"> Java 2D Graph
    //     * Package 2.4</A>, which in turn is a port from the <A
    //     * HREF="http://people.ne.mediaone.net/moshier/index.html#Cephes">Cephes
    //     * 2.2</A> Math Library (C).
    //     * 
    //     * @param a the argument to the function.
    
    {
        double y;
        double z;
        std::vector<double> T{9.60497373987051638749E0, 9.00260197203842689217E1,
        2.23200534594684319226E3, 7.00332514112805075473E3,
        5.55923013010394962768E4};
        std::vector<double> U{3.35617141647503099647E1, 5.21357949780152679795E2,
        4.59432382970980127987E3, 2.26290000613890934246E4,
        4.92673942608635921086E4};

        if (abs(x) > 1.0)
        {
            return (1.0 - errorFunctionComplemented(x));
        }
        z = x * x;
        y = x * polevl(z, T, 4) / p1evl(z, U, 5);
        return y;
    }

    static double errorFunctionComplemented(double a)
    // *
    //     * Returns the complementary Error function of the normal distribution.
    //     * 
    //     * <pre>
    //     *  1 - erf(x) =
    //     * 
    //     *                           inf. 
    //     *                             -
    //     *                  2         | |          2
    //     *   erfc(x)  =  --------     |    exp( - t  ) dt
    //     *               sqrt(pi)   | |
    //     *                           -
    //     *                            x
    //     * </pre>
    //     * 
    //     * <b>Implementation:</b> For small x, <tt>erfc(x) = 1 - erf(x)</tt>;
    //     * otherwise rational approximations are computed.
    //     * <p>
    //     * Code adapted from the <A
    //     * HREF="http://www.sci.usq.edu.au/staff/leighb/graph/Top.html"> Java 2D Graph
    //     * Package 2.4</A>, which in turn is a port from the <A
    //     * HREF="http://people.ne.mediaone.net/moshier/index.html#Cephes">Cephes
    //     * 2.2</A> Math Library (C).
    //     * 
    //     * @param a the argument to the function.


    
    {
        double x;
        double y;
        double z;
        double p;
        double q;
        std::vector<double> P{2.46196981473530512524E-10, 5.64189564831068821977E-1,
        7.46321056442269912687E0, 4.86371970985681366614E1,
        1.96520832956077098242E2, 5.26445194995477358631E2,
        9.34528527171957607540E2, 1.02755188689515710272E3,
        5.57535335369399327526E2};
        std::vector<double> Q{1.32281951154744992508E1, 8.67072140885989742329E1,
        3.54937778887819891062E2, 9.75708501743205489753E2,
        1.82390916687909736289E3, 2.24633760818710981792E3,
        1.65666309194161350182E3, 5.57535340817727675546E2};
        std::vector<double> R{5.64189583547755073984E-1, 1.27536670759978104416E0,
        5.01905042251180477414E0, 6.16021097993053585195E0,
        7.40974269950448939160E0, 2.97886665372100240670E0};
        std::vector<double> S{2.26052863220117276590E0, 9.39603524938001434673E0,
        1.20489539808096656605E1, 1.70814450747565897222E1,
        9.60896809063285878198E0, 3.36907645100081516050E0};

        if (a < 0.0)
        {
            x = -a;
        }
        else 
        {
            x = a;
        }

        if (x < 1.0)
        {
            return 1.0 - errorFunction(a);
        }

        z = -a * a;

        if (z < - MAXLOG)
        {
            if (a < 0)
            {
                return (2.0);
            }
            else 
            {
                return (0.0);
            }
        }

        z = exp(z);

        if (x < 8.0)
        {
            p = polevl(x, P, 8);
            q = p1evl(x, Q, 8);
        }
        else 
        {
            p = polevl(x, R, 5);
            q = p1evl(x, S, 6);
        }

        y = (z * p) / q;

        if (a < 0)
        {
            y = 2.0 - y;
        }

        if (y == 0.0)
        {
            if (a < 0)
            {
                return 2.0;
            }
            else 
            {
                return (0.0);
            }
        }
        return y;
    }

    static double p1evl(double x, std::vector<double> &coef, int N)
    // *
    //     * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
    //     * Evaluates polynomial when coefficient of N is 1.0. Otherwise same as
    //     * <tt>polevl()</tt>.
    //     * 
    //     * <pre>
    //     *                     2          N
    //     * y  =  C  + C x + C x  +...+ C x
    //     *        0    1     2          N
    //     * 
    //     * Coefficients are stored in reverse order:
    //     * 
    //     * coef[0] = C  , ..., coef[N] = C  .
    //     *            N                   0
    //     * </pre>
    //     * 
    //     * The function <tt>p1evl()</tt> assumes that <tt>coef[N] = 1.0</tt> and is
    //     * omitted from the array. Its calling arguments are otherwise the same as
    //     * <tt>polevl()</tt>.
    //     * <p>
    //     * In the interest of speed, there are no checks for out of bounds arithmetic.
    //     * 
    //     * @param x argument to the polynomial.
    //     * @param coef the coefficients of the polynomial.
    //     * @param N the degree of the polynomial.
    {
        double ans;
        ans = x + coef[0];
        
        for (int i = 1; i < N; i++)
        {
            ans = ans * x + coef[i];
        }
        
        return ans;
    }

    static double polevl(double x, std::vector<double> coef, int N)
    // *
    //     * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
    //     * 
    //     * <pre>
    //     *                     2          N
    //     * y  =  C  + C x + C x  +...+ C x
    //     *        0    1     2          N
    //     * 
    //     * Coefficients are stored in reverse order:
    //     * 
    //     * coef[0] = C  , ..., coef[N] = C  .
    //     *            N                   0
    //     * </pre>
    //     * 
    //     * In the interest of speed, there are no checks for out of bounds arithmetic.
    //     * 
    //     * @param x argument to the polynomial.
    //     * @param coef the coefficients of the polynomial.
    //     * @param N the degree of the polynomial.
    
    {
        double ans;
        ans = coef[0];
        
        for (int i = 1; i <= N; i++)
        {
            ans = ans * x + coef[i];
        }
        
        return ans;
    }
    
    static double incompleteGamma(double a, double x)
    // *
    //     * Returns the Incomplete Gamma function.
    //     * 
    //     * @param a the parameter of the gamma distribution.
    //     * @param x the integration end point.
    
    {
        double ans;
        double ax;
        double c;
        double r;

        if (x <= 0 || a <= 0)
        {
            return 0.0;
        }

        if (x > 1.0 && x > a)
        {
            return 1.0 - incompleteGammaComplement(a, x);
        }

        // Compute x**a * exp(-x) / gamma(a) 
        ax = a * log(x) - x - lnGamma(a);
        if (ax < - MAXLOG)
        {
            return (0.0);
        }

        ax = exp(ax);

        // power series 
        r = a;
        c = 1.0;
        ans = 1.0;

        do
        {
            r += 1.0;
            c *= x / r;
            ans += c;
        } while (c / ans > MACHEP);

        return (ans * ax / a);
    }

    static double incompleteGammaComplement(double a, double x)
    // *
    //     * Returns the Complemented Incomplete Gamma function.
    //     * 
    //     * @param a the parameter of the gamma distribution.
    //     * @param x the integration start point.
    
    {
        double ans;
        double ax;
        double c;
        double yc;
        double r;
        double t;
        double y;
        double z;

        double pk;
        double pkm1;
        double pkm2;
        double qk;
        double qkm1;
        double qkm2;

        if (x <= 0 || a <= 0)
        {
            return 1.0;
        }

        if (x < 1.0 || x < a)
        {
            return 1.0 - incompleteGamma(a, x);
        }

        ax = a * log(x) - x - lnGamma(a);
        if (ax < - MAXLOG)
        {
            return 0.0;
        }

        ax = exp(ax);

        // continued fraction 
        y = 1.0 - a;
        z = x + y + 1.0;
        c = 0.0;
        pkm2 = 1.0;
        qkm2 = x;
        pkm1 = x + 1.0;
        qkm1 = z * x;
        ans = pkm1 / qkm1;

        do
        {
            c += 1.0;
            y += 1.0;
            z += 2.0;
            yc = y * c;
            pk = pkm1 * z - pkm2 * yc;
            qk = qkm1 * z - qkm2 * yc;
            if (qk != 0)
            {
                r = pk / qk;
                t = abs((ans - r) / r);
                ans = r;
            }
            else 
            {
                t = 1.0;
            }

            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            if (abs(pk) > big)
            {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
        } while (t > MACHEP);

        return ans * ax;
    }

    static double gamma(double x)
    // *
    //     * Returns the Gamma function of the argument.
    
    {
        std::vector<double> P{1.60119522476751861407E-4, 1.19135147006586384913E-3,
        1.04213797561761569935E-2, 4.76367800457137231464E-2,
        2.07448227648435975150E-1, 4.94214826801497100753E-1,
        9.99999999999999996796E-1};
        std::vector<double> Q{-2.31581873324120129819E-5, 5.39605580493303397842E-4,
        -4.45641913851797240494E-3, 1.18139785222060435552E-2,
        3.58236398605498653373E-2, -2.34591795718243348568E-1,
        7.14304917030273074085E-2, 1.00000000000000000320E0};
        
        double p;
        double z;
        double q = abs(x);
        
        if (q > 33.0)
        {
            if (x < 0.0)
            {
                p = floor(q);
                if (p == q)
                {
                    throw std::logic_error("gamma: overflow");
                }
                z = q - p;
                if (z > 0.5)
                {
                    p += 1.0;
                    z = q - p;
                }
                z = q * sin(M_PI * z);
                if (z == 0.0)
                {
                    throw std::logic_error("gamma: overflow");
                }
                z = abs(z);
                z = M_PI / (z * stirlingFormula(q));
                
                return -z;
            }
            else 
            {
                return stirlingFormula(x);
            }
        }

        z = 1.0;
        while (x >= 3.0)
        {
            x -= 1.0;
            z *= x;
        }

        while (x < 0.0)
        {
            if (x == 0.0)
            {
                throw std::logic_error("gamma: singular");
            }
            else if (x > -1.0E-9)
            {
                return (z / ((1.0 + 0.5772156649015329 * x) * x));
            }
            z /= x;
            x += 1.0;
        }

        while (x < 2.0)
        {
            if (x == 0.0)
            {
                throw std::logic_error("gamma: singular");
            }
            else if (x < 1.0E-9)
            {
                return (z / ((1.0 + 0.5772156649015329 * x) * x));
            }
            z /= x;
            x += 1.0;
        }

        if ((x == 2.0) || (x == 3.0))
        {
            return z;
        }

        x -= 2.0;

        p = polevl(x, P, 6);
        q = polevl(x, Q, 7);
        return z * p / q;
    }

    static double stirlingFormula(double x)
    // *
    //     * Returns the Gamma function computed by Stirling's formula. The polynomial
    //     * STIR is valid for 33 <= x <= 172.
    
    {
        std::vector<double> STIR{7.87311395793093628397E-4, -2.29549961613378126380E-4,
        -2.68132617805781232825E-3, 3.47222221605458667310E-3,
        8.33333333333482257126E-2,};
        double MAXSTIR = 143.01608;

        double w = 1.0 / x;
        double y = exp(x);

        w = 1.0 + w * polevl(w, STIR, 4);

        if (x > MAXSTIR)
        {
            // Avoid overflow in Math.pow() 
            double v = pow(x,0.5 * x - 0.25);
            y = v * (v / y);
        }
        else 
        {
            y = pow(x,x - 0.5) / y;
        }
        y = SQTPI * y * w;
        return y;
    }

    static double incompleteBeta(double aa, double bb, double xx)
    // *
    //     * Returns the Incomplete Beta Function evaluated from zero to <tt>xx</tt>.
    //     * 
    //     * @param aa the alpha parameter of the beta distribution.
    //     * @param bb the beta parameter of the beta distribution.
    //     * @param xx the integration end point.
    
    {
        double a;
        double b;
        double t;
        double x;
        double xc;
        double w;
        double y;
        bool flag;

        if (aa <= 0.0 || bb <= 0.0)
        {
            throw std::logic_error("ibeta: Domain error!");
        }

        if ((xx <= 0.0) || (xx >= 1.0))
        {
            if (xx == 0.0)
            {
                return 0.0;
            }
            if (xx == 1.0)
            {
                return 1.0;
            }
            throw std::logic_error("ibeta: Domain error!");
        }

        flag = false;
        if ((bb * xx) <= 1.0 && xx <= 0.95)
        {
            t = powerSeries(aa, bb, xx);
            return t;
        }

        w = 1.0 - xx;

        // Reverse a and b if x is greater than the mean.
        if (xx > (aa / (aa + bb)))
        {
            flag = true;
            a = bb;
            b = aa;
            xc = xx;
            x = w;
        }
        else 
        {
            a = aa;
            b = bb;
            xc = w;
            x = xx;
        }

        if (flag && (b * x) <= 1.0 && x <= 0.95)
        {
            t = powerSeries(a, b, x);
            if (t <= MACHEP)
            {
                t = 1.0 - MACHEP;
            }
            else 
            {
                t = 1.0 - t;
            }
            return t;
        }
        
        // Choose expansion for better convergence. 
        y = x * (a + b - 2.0) - (a - 1.0);
        if (y < 0.0)
        {
            w = incompleteBetaFraction1(a, b, x);
        }
        else 
        {
            w = incompleteBetaFraction2(a, b, x) / xc;
        }

        //       * Multiply w by the factor a b _ _ _ x (1-x) | (a+b) / ( a | (a) | (b) ) .
        
        y = a * log(x);
        t = b * log(xc);
        if ((a + b) < MAXGAM && abs(y) < MAXLOG && abs(t) < MAXLOG)
        {
            t = pow(xc,b);
            t *= pow(x,a);
            t /= a;
            t *= w;
            t *= gamma(a + b) / (gamma(a) * gamma(b));
            if (flag)
            {
                if (t <= MACHEP)
                {
                    t = 1.0 - MACHEP;
                }
                else 
                {
                    t = 1.0 - t;
                }
            }
            return t;
        }
        // Resort to logarithms.
        y += t + lnGamma(a + b) - lnGamma(a) - lnGamma(b);
        y += log(w / a);
        if (y < MINLOG)
        {
            t = 0.0;
        }
        else 
        {
            t = exp(y);
        }
        if (flag)
        {
            if (t <= MACHEP)
            {
                t = 1.0 - MACHEP;
            }
            else 
            {
                t = 1.0 - t;
            }
        }
        return t;
    }

    static double incompleteBetaFraction1(double a, double b, double x)
    // *
    //     * Continued fraction expansion #1 for incomplete beta integral.
    {
        double xk;
        double pk;
        double pkm1;
        double pkm2;
        double qk;
        double qkm1;
        double qkm2;
        double k1;
        double k2;
        double k3;
        double k4;
        double k5;
        double k6;
        double k7;
        double k8;
        double r;
        double t;
        double ans;
        double thresh;
        int n;

        k1 = a;
        k2 = a + b;
        k3 = a;
        k4 = a + 1.0;
        k5 = 1.0;
        k6 = b - 1.0;
        k7 = k4;
        k8 = a + 2.0;

        pkm2 = 0.0;
        qkm2 = 1.0;
        pkm1 = 1.0;
        qkm1 = 1.0;
        ans = 1.0;
        r = 1.0;
        n = 0;
        thresh = 3.0 * MACHEP;
        do
        {
            xk = -(x * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            xk = (x * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if (qk != 0)
            {
                r = pk / qk;
            }
            if (r != 0)
            {
                t = abs((ans - r) / r);
                ans = r;
            }
            else 
            {
                t = 1.0;
            }
            
            if (t < thresh)
            {
                return ans;
            }

            k1 += 1.0;
            k2 += 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 -= 1.0;
            k7 += 2.0;
            k8 += 2.0;

            if ((abs(qk) + abs(pk)) > big)
            {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
            if ((abs(qk) < biginv) || (abs(pk) < biginv))
            {
                pkm2 *= big;
                pkm1 *= big;
                qkm2 *= big;
                qkm1 *= big;
            }
        } while (++n < 300);
        return ans;
    }
    static double incompleteBetaFraction2(double a, double b, double x)
    // *
    //     * Continued fraction expansion #2 for incomplete beta integral.
    {
        double xk;
        double pk;
        double pkm1;
        double pkm2;
        double qk;
        double qkm1;
        double qkm2;
        double k1;
        double k2;
        double k3;
        double k4;
        double k5;
        double k6;
        double k7;
        double k8;
        double r;
        double t;
        double ans;
        double z;
        double thresh;
        int n;

        k1 = a;
        k2 = b - 1.0;
        k3 = a;
        k4 = a + 1.0;
        k5 = 1.0;
        k6 = a + b;
        k7 = a + 1.0;
        ;
        k8 = a + 2.0;

        pkm2 = 0.0;
        qkm2 = 1.0;
        pkm1 = 1.0;
        qkm1 = 1.0;
        z = x / (1.0 - x);
        ans = 1.0;
        r = 1.0;
        n = 0;
        thresh = 3.0 * MACHEP;
        do
        {
            xk = -(z * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            xk = (z * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if (qk != 0)
            {
                r = pk / qk;
            }
            if (r != 0)
            {
                t = abs((ans - r) / r);
                ans = r;
            }
            else 
            {
                t = 1.0;
            }

            if (t < thresh)
            {
                return ans;
            }

            k1 += 1.0;
            k2 -= 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 += 1.0;
            k7 += 2.0;
            k8 += 2.0;

            if ((abs(qk) + abs(pk)) > big)
            {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
            if ((abs(qk) < biginv) || (abs(pk) < biginv))
            {
                pkm2 *= big;
                pkm1 *= big;
                qkm2 *= big;
                qkm1 *= big;
            }
        } while (++n < 300);
        return ans;
    }
    static double powerSeries(double a, double b, double x)
    // *
    // * Power series for incomplete beta integral. Use when b*x is small and x not
    // * too close to 1.
    {
        double s;
        double t;
        double u;
        double v;
        double n;
        double t1;
        double z;
        double ai;

        ai = 1.0 / a;
        u = (1.0 - b) * x;
        v = u / (a + 1.0);
        t1 = v;
        t = u;
        n = 2.0;
        s = 0.0;
        z = MACHEP * ai;
        while (abs(v) > z)
        {
            u = (n - b) * x / n;
            t *= u;
            v = t / (a + n);
            s += v;
            n += 1.0;
        }
        s += t1;
        s += ai;
        
        u = a * log(x);
        
        if ((a + b) < MAXGAM && abs(u) < MAXLOG)
        {
            t = gamma(a + b) / (gamma(a) * gamma(b));
            s = s * t * pow(x,a);
        }
        else 
        {
            t = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + u + log(s);
            if (t < MINLOG)
            {
                s = 0.0;
            }
            else 
            {
                s = exp(t);
            }
        }
        return s;
    }
};
#endif

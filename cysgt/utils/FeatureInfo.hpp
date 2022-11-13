#ifndef FEATUREINFO
#define FEATUREINFO

#include "FeatureType.hpp"

class FeatureInfo
{

public:
    FeatureType type;
    int categories;

    void setFeatureType(FeatureType t)
    {
        type = t;
    }

    FeatureType getFeatureType()
    {
        return type;
    }

    void setCategories(int c)
    {
        categories = c;
    }

    int getCategories()
    {
        return categories;
    }
};

#endif
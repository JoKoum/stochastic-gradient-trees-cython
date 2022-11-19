from libcpp.vector cimport vector
import numpy as np

cdef extern from "utils/StochasticGradientTree.hpp":
    cdef cppclass StochasticGradientTree:
        StochasticGradientTree(char*, int, int, int, double, double, vector[double], vector[double], double)
        int getEpochs()
        void setEpochs(int)
        int getBins()
        void setBins(int)
        void setFit(int)
        int getFit()
        void setTrainBatchSize(int)
        int getTrainBatchSize()
        void setLambda(double)
        double getLambda()
        void setGamma(double)
        double getGamma()
        int getDepth()
        int getTotalNodes()
        void setLearningRate(double)
        void setBounds(vector[double], vector[double])
        int getIsFit()
        int getClassifierType()
        vector[vector[int]] createFeatures(vector[vector[double]], vector[double], vector[double])
        void train(vector[int], double)
        void fit(vector[vector[double]], vector[double])
        vector[vector[double]] predictProba(vector[vector[double]])


cdef class PyStochasticGradientTree:
    cdef StochasticGradientTree* thisptr
    cdef char* ob
    cdef int binNo
    cdef int batch_size 
    cdef int epochNo 
    cdef double l 
    cdef double g 
    cdef vector[double] upper 
    cdef vector[double] lower 
    cdef double lr

    def __init__(self, char* ob, int binNo = 64, int batch_size = 200, int epochNo = 20, double l = 0.1, double g = 1.0, vector[double] upper = vector[double](), vector[double] lower = vector[double](), double lr = 1.0):
        self.thisptr = new StochasticGradientTree(ob, binNo, batch_size, epochNo, l, g, upper, lower, lr)
        self.ob = ob 
        self.binNo = binNo
        self.batch_size = batch_size
        self.epochNo = epochNo
        self.l = l
        self.g = g
        self.upper = upper
        self.lower = lower
        self.lr = lr

    def setEpochs(self, int d):
        self.thisptr.setEpochs(d)
    
    def getEpochs(self):
        return self.thisptr.getEpochs()

    def setBins(self, int b):
        self.thisptr.setBins(b)
    
    def getBins(self):
        return self.thisptr.getBins()

    def setTrainBatchSize(self, int bs):
        self.thisptr.setTrainBatchSize(bs)

    def getTrainBatchSize(self):
        return self.thisptr.getTrainBatchSize()

    def setLambda(self, double l):
        self.thisptr.setLambda(l)
    
    def getLambda(self):
        return self.thisptr.getLambda()

    def setGamma(self, double g):
        self.thisptr.setGamma(g)

    def getGamma(self):
        return self.thisptr.getGamma()

    def get_depth(self):
        return self.thisptr.getDepth()

    def get_total_nodes(self):
        return self.thisptr.getTotalNodes()

    def set_learning_rate(self, double lr):
        self.thisptr.setLearningRate(lr)

    def is_fit(self):
        return self.thisptr.getIsFit() 
        
    def fit(self, X, y):
        if hasattr(X, "dtypes") and hasattr(X, "__array__"):
            X = X.to_numpy()

        isFit = self.thisptr.getFit()
        if isFit == 0:
            upper_bounds = np.max(X, axis=0).tolist()
            lower_bounds = np.min(X, axis=0).tolist()
            self.thisptr.setBounds(upper_bounds, lower_bounds)

        X = X.tolist()

        if 'pandas' in str(type(y)):
            y = y.to_numpy().tolist()

        self.thisptr.fit(X, y)
    
    def predict_proba(self, X):
        if hasattr(X, "dtypes") and hasattr(X, "__array__"):
            X = X.to_numpy()

        isFit = self.thisptr.getFit()
        if isFit == 0:
            upper_bounds = np.max(X, axis=0).tolist()
            lower_bounds = np.min(X, axis=0).tolist()
            self.thisptr.setBounds(upper_bounds, lower_bounds)
            
        X = X.tolist()
        return np.array(self.thisptr.predictProba(X))

    def predict(self, X):        
        y_pred = self.predict_proba(X)

        classifier = self.thisptr.getClassifierType()

        if classifier == 0:
            return [np.argmax(pred) for pred in y_pred]
        elif classifier == 1:
            return [np.max(pred) for pred in y_pred]

    def __copy__(self):
        return PyStochasticGradientTree(self.ob, self.binNo, self.batch_size, self.epochNo, self.l, self.g, self.upper, self.lower, self.lr)

    def __dealloc__(self):
        if not self.thisptr == NULL:
            del self.thisptr
    
    def __reduce__(self):
        return (self.__class__, (self.ob, self.binNo, self.batch_size, self.epochNo, self.l, self.g, self.upper, self.lower, self.lr))

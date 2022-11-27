from libcpp.vector cimport vector
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
from libcpp.string cimport string

cdef extern from "utils/StochasticGradientTree.hpp":
    cdef cppclass StochasticGradientTree:
        string obType
        int bins
        int batchSize
        int epochs
        double mLambda
        double gamma
        vector[double] upper_bounds
        vector[double] lower_bounds
        double learning_rate
        StochasticGradientTree(string, int, int, int, double, double, vector[double], vector[double], double)
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
    cpdef StochasticGradientTree *_thisptr
    cpdef string ob
    cpdef int binNo
    cpdef int batch_size 
    cpdef int epochNo 
    cpdef double l 
    cpdef double g 
    cpdef vector[double] upper 
    cpdef vector[double] lower 
    cpdef double lr

    def __init__(self, string ob, int binNo = 64, int batch_size = 200, int epochNo = 20, double l = 0.1, double g = 1.0, vector[double] upper = vector[double](), vector[double] lower = vector[double](), double lr = 1.0):
        self._thisptr = new StochasticGradientTree(ob, binNo, batch_size, epochNo, l, g, upper, lower, lr)
        self.ob = ob 
        self.binNo = binNo
        self.batch_size = batch_size
        self.epochNo = epochNo
        self.l = l
        self.g = g
        self.upper = upper
        self.lower = lower
        self.lr = lr

    cpdef bytes get_data(self):
        if self._thisptr == NULL:
            return None
        return <bytes>(<char *>self._thisptr)

    cpdef void set_data(self, bytes thisptr):
        PyMem_Free(self._thisptr)
        self._thisptr = <StochasticGradientTree*>PyMem_Malloc(sizeof(StochasticGradientTree))
        if not self._thisptr:
            raise MemoryError()
        memcpy(self._thisptr, <char *>thisptr, sizeof(StochasticGradientTree))

    property thisptr:
        def __get__(self):    
            return [(self._thisptr.obType, self._thisptr.bins,
                self._thisptr.batchSize, self._thisptr.epochs,
                self._thisptr.mLambda, self._thisptr.gamma,
                self._thisptr.upper_bounds, self._thisptr.lower_bounds,
                self._thisptr.learning_rate)]
        def __set__(self, values):
            self._thisptr = <StochasticGradientTree*>PyMem_Malloc(sizeof(StochasticGradientTree))
            if not self._thisptr:
                raise MemoryError()
            ob, binNo, batch_size, epochNo, l, g, upper, lower, lr = values
            self._thisptr.obType = ob
            self._thisptr.bins = binNo
            self._thisptr.batchSize = batch_size
            self._thisptr.epochs = epochNo
            self._thisptr.mLambda = l
            self._thisptr.gamma = g
            self._thisptr.upper_bounds = upper
            self._thisptr.lower_bounds = lower
            self._thisptr.learning_rate = lr
    
    def setEpochs(self, int d):
        self._thisptr.setEpochs(d)
    
    def getEpochs(self):
        return self._thisptr.getEpochs()

    def setBins(self, int b):
        self._thisptr.setBins(b)
    
    def getBins(self):
        return self._thisptr.getBins()

    def setTrainBatchSize(self, int bs):
        self._thisptr.setTrainBatchSize(bs)

    def getTrainBatchSize(self):
        return self._thisptr.getTrainBatchSize()

    def setLambda(self, double l):
        self._thisptr.setLambda(l)
    
    def getLambda(self):
        return self._thisptr.getLambda()

    def setGamma(self, double g):
        self._thisptr.setGamma(g)

    def getGamma(self):
        return self._thisptr.getGamma()

    def get_depth(self):
        return self._thisptr.getDepth()

    def get_total_nodes(self):
        return self._thisptr.getTotalNodes()

    def set_learning_rate(self, double lr):
        self._thisptr.setLearningRate(lr)

    def is_fit(self):
        return self._thisptr.getIsFit() 
        
    def fit(self, X, y):
        if hasattr(X, "dtypes") and hasattr(X, "__array__"):
            X = X.to_numpy()

        isFit = self._thisptr.getFit()
        if isFit == 0:
            upper_bounds = np.max(X, axis=0).tolist()
            lower_bounds = np.min(X, axis=0).tolist()
            self._thisptr.setBounds(upper_bounds, lower_bounds)

        X = X.tolist()

        if 'pandas' in str(type(y)):
            y = y.to_numpy().tolist()

        self._thisptr.fit(X, y)
    
    def predict_proba(self, X):
        if hasattr(X, "dtypes") and hasattr(X, "__array__"):
            X = X.to_numpy()

        isFit = self._thisptr.getFit()
        if isFit == 0:
            upper_bounds = np.max(X, axis=0).tolist()
            lower_bounds = np.min(X, axis=0).tolist()
            self._thisptr.setBounds(upper_bounds, lower_bounds)
            
        X = X.tolist()
        return np.array(self._thisptr.predictProba(X))

    def predict(self, X):        
        y_pred = self.predict_proba(X)

        classifier = self._thisptr.getClassifierType()

        if classifier == 0:
            return [np.argmax(pred) for pred in y_pred]
        elif classifier == 1:
            return [np.max(pred) for pred in y_pred]

    def __copy__(self):
        return PyStochasticGradientTree(self.ob, self.binNo, self.batch_size, self.epochNo, self.l, self.g, self.upper, self.lower, self.lr)

    def __dealloc__(self):
        PyMem_Free(self._thisptr)

    def __getstate__(self):
        return self.get_data()

    def __setstate__(self, state):
        self.set_data(state)

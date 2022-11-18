from SGTWrapper import PyStochasticGradientTree as StochasticGradientTree
from sklearn.base import BaseEstimator


class SGTClassifier(StochasticGradientTree):
    def __init__(self, objective=b"classification", bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0, upper_bounds=[], lower_bounds=[], learning_rate=1.0):
        super().__init__(
            ob=objective,
            binNo=bins,
            batch_size=batch_size,
            epochNo=epochs,
            l=m_lambda,
            g=gamma,
            upper=upper_bounds,
            lower=lower_bounds,
            lr=learning_rate
            )
        self._estimator_type = 'classifier'
        self.objective=objective
        self.bins=bins
        self.batch_size=batch_size
        self.epochs=epochs
        self.m_lambda=m_lambda
        self.gamma=gamma
        self.upper_bounds=upper_bounds
        self.lower_bounds=lower_bounds
        self.lr = learning_rate
    

    def __copy__(self):
        return SGTClassifier(self.objective, self.bins, self.batch_size, self.epochs, self.m_lambda, self.gamma, self.upper_bounds, self.lower_bounds, self.lr)

  
class SGTRegressor(StochasticGradientTree):
    def __init__(self, objective=b"regression", bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1.0, upper_bounds=[], lower_bounds=[], learning_rate=1.0):
        super().__init__(
            ob=objective,
            binNo=bins,
            batch_size=batch_size,
            epochNo=epochs,
            l=m_lambda,
            g=gamma,
            upper=upper_bounds,
            lower=lower_bounds,
            lr=learning_rate
            )
        self._estimator_type = 'regressor'
        self.objective=objective
        self.bins=bins
        self.batch_size=batch_size
        self.epochs=epochs
        self.m_lambda=m_lambda
        self.gamma=gamma
        self.upper_bounds=upper_bounds
        self.lower_bounds=lower_bounds
        self.lr = learning_rate

    
    def __copy__(self):
        return SGTRegressor(self.objective, self.bins, self.batch_size, self.epochs, self.m_lambda, self.gamma, self.upper_bounds, self.lower_bounds, self.lr)


class StochasticGradientTreeClassifier(SGTClassifier, BaseEstimator):
    def __init__(self, objective=b"classification", bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1, upper_bounds=[], lower_bounds=[]):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            learning_rate=1.0
            )
        self.objective=objective
        self.bins=bins
        self.batch_size=batch_size
        self.epochs=epochs
        self.m_lambda=m_lambda
        self.gamma=gamma
        self.upper_bounds=upper_bounds
        self.lower_bounds=lower_bounds


class StochasticGradientTreeRegressor(SGTRegressor, BaseEstimator):
    def __init__(self, objective=b"regression", bins=64, batch_size=200, epochs=20, m_lambda=0.1, gamma=1, upper_bounds=[], lower_bounds=[]):
        super().__init__(
            objective=objective,
            bins=bins,
            batch_size=batch_size,
            epochs=epochs,
            m_lambda=m_lambda,
            gamma=gamma,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            learning_rate=1.0
            )
        self.objective=objective
        self.bins=bins
        self.batch_size=batch_size
        self.epochs=epochs
        self.m_lambda=m_lambda
        self.gamma=gamma
        self.upper_bounds=upper_bounds
        self.lower_bounds=lower_bounds
    
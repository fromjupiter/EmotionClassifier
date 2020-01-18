from sklearn.preprocessing import OneHotEncoder
from dataloader import load_data, balanced_sampler
import numpy as np
import random
from numpy import matlib


class SoftmaxRegression(object):
    #
    # denote sample number as n, data dimension as m, label number as k
    #
    def __init__(self, lr=0.01, iter_times=100, class_weight=None):
        self.iter_times = iter_times

        self.rate = lr
        # m*k
        self.coef_ = None
        self.enc_ = None
        self.class_weight = class_weight
        self.classes_ = None
        self.weights_ = None
    
    def setup(self, X, y):
        y = self._encode_y(y)
        if self.coef_ is None:
            self.coef_ = np.zeros((X.shape[1], len(self.classes_)))
        if self.weights_ is None:
            # always do balanced weight for now
            self.weights_ = np.array(len(y) / len(self.classes_) / y.sum(axis=0)).flatten()
        return y
        
        
    def _encode_y(self, y):
        if type(y)==np.ndarray or type(y)==list:
            y = np.array(y).reshape(-1,1)
            if self.enc_ is None:
                self.enc_ = OneHotEncoder(handle_unknown='ignore')
                self.enc_.fit(y)
                self.classes_ = self.enc_.categories_[0]
            return  self.enc_.transform(y).todense()
        return y

    # average cross entropy, y is string array
    def _loss(self, X, y):
        y = self._encode_y(y)
        pred = self.predict_proba(X)
        # return -np.sum(np.multiply(y, np.log(pred)))/len(X)
        return -np.sum(np.multiply(y, np.log(pred)).dot(self.weights_))/len(X)

    # t: encoded labels
    def fit_one_epoch(self, X, y, use_batch=True):
        y = self.setup(X, y)
        if use_batch:
            # gd
            delta = self.predict_proba(X) - y
            grad = X.T.dot(delta) / len(X)
            self.coef_ -= self.rate * grad
        else:
            # sgd
            indices = list(range(0, len(X)))
            random.shuffle(indices)
            for i in indices:
                delta = self.predict_proba(X[i]) - y[i]
                grad = X[i].T.dot(delta)
                self.coef_ -= self.rate * grad

    def fit(self, X, y):
        i = 0
        while i < self.iter_times:
            self.fit_one_epoch(X, y)
            i+=1

    def predict(self, X):
        y = self.predict_proba(X).argmax(axis=1)
        return np.array(self.classes_[y]).flatten()

    def predict_proba(self, X):
        # a is n*k: (n*m).(m*k)
        a = X.dot(self.coef_)
        # prob:n*k
        a = np.exp(a)
        return a/a.sum(axis=1)


if __name__ == '__main__':
    X_train = matlib.ones((100,5))
    y_train = np.array([0]*50 + [1]*50)

    regressor = SoftmaxRegression()
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_train)
    
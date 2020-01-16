import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MyLogisticRegression:
    def __init__(self, lr):
        self.rate = lr
        self.coef_ = None

    def encode_Y(self,Y):
        if Y.dtype == 'int32':
            return Y
        elif Y.dtype == 'U1':
            Y = Y.tolist()
            b = defaultdict(lambda : len(b))
            Y = np.array(list(map(lambda x:b[x],Y)))
            return Y
        else:
            print('not supporting')

    def sigmoid(self,data):
        return 1/(1+np.exp(-np.dot(data,self.coef_)))

    def init_w(self,components):
        self.coef_ = np.zeros((components,1))

    def gradient_decent(self,predict_label,X,Y):
        m = X.shape[0]
        grad = 1/m*np.dot(X.T,(predict_label-Y))
        self.coef_= self.coef_-self.rate*grad

    def _loss(self,X,Y):
        pred = self.predict(X)
        m = Y.shape[0]
        loss = -1/m* np.sum(Y*np.log(pred)+(1-Y)*np.log(1-pred))
        return loss

    def fit_one_epoch(self,X,Y,stochastic = False):
        Y =self.encode_Y(Y)
        if type(self.coef_) != 'numpy.ndarray':
            self.init_w(X.shape[1])
        if not stochastic:
            predict_label = self.sigmoid(X)
            self.gradient_decent(predict_label,X,Y)
        else:
            for i in range(X.shape[0]):
                predict_label = self.sigmoid(X[i:i+1,:])
                self.gradient_decent(predict_label,X[i:i+1,:],Y[i:i+1])               

        

    def predict(self,X):
        return self.sigmoid(X)

    def fit(self,X,Y,epoch):
        # components = X.shape[1]
        # Y =self.encode_Y(Y)
        # self.init_w(components)
        for i in range(epoch):
            self.fit_one_epoch(X,Y)
        return self.coef_


if __name__ == '__main__':
    X = np.array([[1,2,3,4],[1,45,6,3],[1,6,7,2]])
    # Y = np.array([0,1,0]).T
    Y = np.array(['a','b','a']).T
    clr = MyLogisticRegression(lr = 0.005)
    # components = X.shape[1]
    # clr.init_w(components)
    # Y = clr.encode_Y(Y)
    for i in range(5):
        clr.fit_one_epoch(X,Y,stochastic = True)

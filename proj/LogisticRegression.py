import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class LogisticRegression:
    def __init__(self, lr,epoch = 1000):
        self.rate = lr
        self.coef_ = None
        self.epoch = epoch
        self.classes_ = None
    def setup(self,X,y):
        self.init_w(X.shape[1])
        
    def _encode_y(self,Y):
        if self.classes_ is None:
            self.classes_ = []
            [self.classes_.append(i) for i in Y if not i in self.classes_]
            self.classes_ = np.array(self.classes_)
        Y = Y.tolist()
        b = defaultdict(lambda : len(b))
        Y = np.matrix(list(map(lambda x:b[x],Y))).T
        return Y

    def sigmoid(self,data):
        return 1/(1+np.exp(-np.dot(data,self.coef_)))

    def init_w(self,components):
        self.coef_ = np.zeros((components,1))

    def gradient_decent(self,predict_label,X,Y):
        m = X.shape[0]        
        grad = 1/m*np.dot(X.T,(predict_label-Y))
        self.coef_= self.coef_-self.rate*grad

    def _loss(self,X,Y):
        pred = self.predict_proba(X)
        Y =self._encode_y(Y)
        m = Y.shape[0]
        loss = -1/m* np.sum(np.sum(np.multiply(Y,np.log(pred))+np.multiply(1-Y,np.log(1-pred))))
        return loss

    def fit_one_epoch(self,X,Y,usebatch = True):
        Y =self._encode_y(Y)
        if self.coef_ is None:
            self.init_w(X.shape[1])
            
        if usebatch:
            predict_label = self.sigmoid(X)
            self.gradient_decent(predict_label,X,Y)
        else:
            for i in range(X.shape[0]):
                predict_label = self.sigmoid(X[i:i+1,:])
                self.gradient_decent(predict_label,X[i:i+1,:],Y[i:i+1])           

        

    def predict_proba(self,X):
        return self.sigmoid(X)

    def predict(self,X):
        prediction = self.predict_proba(X)
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        prediction = prediction.astype(int)
        return np.array(self.classes_[prediction]).flatten()

    def fit(self,X,Y):

        for i in range(self.epoch):
            self.fit_one_epoch(X,Y)
        return self.coef_


if __name__ == '__main__':
    X = np.array([[1,2,3,4],[1,45,6,3],[1,6,7,2]])
    test = np.array([[1,0,0,0],[1,0,6,2],[1,1000,1,5]])
    # Y = np.array([0,1,0]).T
    Y = np.array(['a','b','a']).T
    clr = LogisticRegression(lr = 0.005)
    # components = X.shape[1]
    # clr.init_w(components)
    # Y = clr._encode_y(Y)
    clr.fit(X,Y)
    print(clr.predict(test))

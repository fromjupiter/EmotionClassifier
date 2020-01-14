import numpy as np
from dataloader import load_data, balanced_sampler
import math

from sklearn.decomposition import PCA

class MyPCA(object):
    def __init__(self, n_components=10):
        self.n_components = n_components
        # temp calc
        self.n_samples = None
        self.mean_ = None
        self.s_values_ = None
        self.components_ = None
    
    # X is M*N matrix, M is sample number, N is data dimension
    def fit(self, X): 
        # A: N*M
        self.n_samples = len(X)
        self.mean_ = X.mean(axis=0)
        A = (X - self.mean_).T
        # L: M*M
        L = A.T.dot(A)/self.n_samples
        eigval, eigvec  = np.linalg.eig(L)
        # select main components
        order = eigval.argsort()[::-1]
        eigval = eigval[order][:self.n_components]
        eigvec = eigvec[:,order][:,:self.n_components]
        
        # Turk and Pentland's trick
        # _components(U): N*k
        self.s_values_ = eigval
        self.components_ = A.dot(eigvec)
        self.components_ /= np.linalg.norm(self.components_, axis=0)
        self.components_ /= np.sqrt(self.s_values_)

    def transform(self, X):
        center = (X - self.mean_)
        X_pca = center.dot(self.components_)
        return X_pca


if __name__ == '__main__':
    data_dir = "./aligned/"
    dataset, cnt = load_data(data_dir)
    # test with happiness and anger
    images = balanced_sampler(dataset, cnt, emotions=['happiness'])['happiness']
    X = np.matrix(list(map(lambda x:x.flatten(), images)))

    pca = MyPCA(n_components=10)
    pca.fit(X)
    X_pca = pca.transform(X)
    # sanity check
    assert np.all(np.abs(X_pca.sum(axis=0))<1e-6)
    assert np.all(np.abs(X_pca.std(axis=0)-1.0)<1e-6)

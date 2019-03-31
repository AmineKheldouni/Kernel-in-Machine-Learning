########################################################################
### Kernel abstract class (with Normalization option)
########################################################################

from imports import *
from scipy.sparse.linalg import norm

from abc import abstractmethod, ABC

class Kernel(ABC):
    def __init__(self, normalize=False):
        self.normalize = normalize
        pass

    @abstractmethod
    def compute_feature_vector(self, X):
        pass

    def compute_train(self, data_train):
        feature_vector = self.compute_feature_vector(data_train)
        K = np.dot(feature_vector, feature_vector.T)
        if self.normalize:
            K = self.normalize_train(K)
        self.K = K
        return K

    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        K = np.dot(feature_vector_test, feature_vector_train.T)
        if self.normalize:
           K = self.normalize_test(K, feature_vector_test)
        return K

    def normalize_train(self, K_train): #K_train unormalized
        self.norms_train = np.sqrt(K_train.diagonal())  # norms for x train vector
        matrix_norms = np.outer(self.norms_train,self.norms_train) #10e-40
        K_train =  np.divide(K_train, matrix_norms)
        self.K_train = K_train
        return K_train

    def normalize_test(self, K_test, feats_test): #K_test unormalized
        m = K_test.shape[0]
        #norms_test = np.sum(feats_test*feats_test,axis=1)
        norms_test = norm(feats_test,axis=1)
        matrix_norms = np.outer(norms_test, self.norms_train) #+ 1e-40  # matrix sqrt(K(xtest,xtest)*K(xtrain,xtrain))
        K_test = np.divide(K_test, matrix_norms)
        return K_test

    def save_train(self, filename):
        np.save(filename, self.K)

    def load_train(self, filename):
        self.K = np.load(filename)

import numpy as np

from abc import abstractmethod, ABC
class FastKernel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_feature_vector(self):
        pass

    def compute_train(self, data_train):
        feature_vector = self.compute_feature_vector(data_train)
        K = np.dot(feature_vector, feature_vector.T)
        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization train:", normalization.shape)
        return K / normalization

    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        K = np.dot(feature_vector_test, feature_vector_train.T)
        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization test:", normalization.shape)
        return K / normalization

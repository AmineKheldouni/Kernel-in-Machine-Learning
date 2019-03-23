########################################################################
### LinearKernel
########################################################################

from kernels.linear_kernel import LinearKernel
import numpy as np


class RBFKernel(LinearKernel):
    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__()

    def compute_train(self, data_train):
        n = len(data_train[0])
        feature_vector = self.compute_feature_vector(data_train)
        return np.exp(- n + self.gamma * np.dot(feature_vector, feature_vector.T))

    def compute_test(self, data_train, data_test):
        n = len(data_train[0])
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        return np.exp(-n + self.gamma * np.dot(feature_vector_test, feature_vector_train.T))
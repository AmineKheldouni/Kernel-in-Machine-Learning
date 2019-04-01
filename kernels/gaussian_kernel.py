########################################################################
### Gaussian kernel on one-hot encoded sequences
########################################################################

from imports import *
from kernels.linear_kernel import LinearKernel

class ExponentialLinearKernel(LinearKernel):
    def __init__(self, gamma, normalize = False):
        self.gamma = gamma
        super().__init__(normalize)

    # ||x-y||² = ||x||² + ||y||² - 2x.y^T
    # As norms are equal (one hot encoding, the following is equivalent to the Gaussian kernel with Euclidian metric
    # on one-hot Euclidian sequences
    def compute_train(self, data_train):
        n = len(data_train[0])
        feature_vector = self.compute_feature_vector(data_train)
        return np.exp(- n + self.gamma * np.dot(feature_vector, feature_vector.T))

    def compute_test(self, data_train, data_test):
        n = len(data_train[0])
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        return np.exp(-n + self.gamma * np.dot(feature_vector_test, feature_vector_train.T))

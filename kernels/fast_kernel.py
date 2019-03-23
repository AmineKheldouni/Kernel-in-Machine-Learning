import numpy as np

class FastKernel:
    def __init__(self):
        pass

    def compute_train(self, data_train):
        feature_vector = self.compute_feature_vector(data_train)
        return np.dot(feature_vector, feature_vector.T)

    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        return np.dot(feature_vector_test, feature_vector_train.T)

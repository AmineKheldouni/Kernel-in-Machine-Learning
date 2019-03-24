from kernels.fast_kernel import FastKernel
import numpy as np
from kernels.fast_spectrum_kernel import SpectrumKernel
import scipy.sparse as ss

class SumSpectrumKernel(FastKernel):
    def __init__(self, list_k):
        super().__init__()
        self.list_k = list_k
        self.index = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.kernels = [SpectrumKernel(k) for k in list_k]

    def compute_index(self, word):
        for i in range(len(word)):
            word[i] = self.index[word[i]]
        return np.dot(np.array(word), np.power(4, np.arange(self.list_k)))

    def compute_feature_vector(self, X):
        # TODO Maybe need sparse matrix for k very large
        features = self.kernels[0].compute_feature_vector(X)
        for i in range(1, len(self.kernels)):
            kernel_feature = self.kernels[i].compute_feature_vector(X)
            features = ss.vstack(([features, kernel_feature])).toarray()
        print("Shape of features from SSK: {}".format(features.shape))
        return features

    def compute_train(self, data_train):
        K = 0
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_train(data_train)

        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization test:", normalization.shape)

        return K/normalization

    def compute_test(self, data_train, data_test):
        K = 0
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_test(data_train, data_test)

        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization test:", normalization.shape)

        return K/normalization

from kernels.fast_kernel import FastKernel
import numpy as np
from kernels.fast_spectrum_kernel import SpectrumKernel

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
            features = np.concatenate((features, kernel_feature), axis = 1)
        print("Shape of features from SSK: {}".format(features.shape))
        return features

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from kernels.fast_kernel import FastKernel

class SpectrumKernel(FastKernel):
    def __init__(self, k, normalize=False):
        super().__init__()
        self.k = k
        self.index = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.normalize = normalize

    def compute_index(self, word):
        for i in range(len(word)):
            word[i] = self.index[word[i]]
        return np.dot(np.array(word), np.power(4, np.arange(self.k)))

    def compute_feature_vector(self, X):
        features =  lil_matrix((X.shape[0], 4 ** self.k))
        for i, line in enumerate(X):
            for j in range(len(line) - self.k + 1):
                features[i, self.compute_index(list(line[j:j + self.k]))] += 1
        return csr_matrix(features)

    def compute_train(self, data_train):
        feature_vector = self.compute_feature_vector(data_train)
        K = np.dot(feature_vector, feature_vector.T).toarray()
        if self.normalize:
            K = self.normalize_train(K)
        return K
        
    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        K = np.dot(feature_vector_test, feature_vector_train.T).toarray()
        if self.normalize:
            K = self.normalize_test(K, feature_vector_test)
        return K

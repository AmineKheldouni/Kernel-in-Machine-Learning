from kernels.fast_kernel import FastKernel
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

class SpectrumKernel():
    def __init__(self, k):
        self.k = k
        self.index = {"A": 0, "C": 1, "G": 2, "T": 3}

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

        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization test:", normalization.shape)

        return K/normalization
        
    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        K = np.dot(feature_vector_test, feature_vector_train.T).toarray()

        normalization = np.sqrt(K.diagonal()).reshape(-1,1).dot(np.sqrt(K.diagonal()).reshape(1,-1))
        print("normalization test:", normalization.shape)

        return K/normalization

from kernels.fast_kernel import FastKernel
import numpy as np

class SpectrumKernel(FastKernel):
    def __init__(self, k):
        self.k = k
        self.index = {"A": 0, "C": 1, "G": 2, "T": 3}
    
    def compute_index(self, word):
        for i in range(len(word)):
            word[i] = self.index[word[i]]
        return np.dot(np.array(word), np.power(4, np.arange(self.k)))
    
    def compute_feature_vector(self, X):
        #TODO Maybe need sparse matrix for k very large
        features =  np.zeros((X.shape[0], 4 ** self.k))
        for i, line in enumerate(X):
            for j in range(len(line) - self.k + 1):
                features[i, self.compute_index(list(line[j:j + self.k]))] += 1 
        return features

from kernels.fast_kernel import FastKernel
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from copy import deepcopy


def memoize(fun):
    memory = {}
    def helper(word, i):
        if (str(word), i) not in memory:
            memory[(str(word), i)] = fun(word, i)
        return memory[(str(word), i)]
    return helper


@memoize
def matches(word, i):
    if i == 0:
        return set(["".join(word)])
    else:
        matches_set = set()
        for j in range(len(word)):
            for letter in ["A", "C", "G", "T"]:
                new_word = deepcopy(word)
                new_word[j] = letter
                matches_set.update(matches(new_word, i-1))
        return matches_set


class MismatchSpectrumKernel(FastKernel):
    def __init__(self, k, m=1, normalize = False):
        super().__init__(normalize)
        self.k = k
        self.m = m
        all_elements = self.all_elements(k)
        self.index = dict(zip(all_elements, np.arange(len(all_elements))))
        self.K_train = None
        self.K_test = None

    def compute_feature_vector(self, X):
        if self.k < 9:
            features =  np.zeros((X.shape[0], 4 ** self.k))
        else:
            features =  lil_matrix((X.shape[0], 4 ** self.k))

        for i, line in enumerate(X):
            for j in range(len(line) - self.k + 1):
                words = matches(list(line[j:j + self.k]), self.m)
                indices = []
                for word in words:
                    indices.append(self.index[word])
                features[i, np.array(indices)] += np.ones(len(indices))
        return csr_matrix(features)

    def compute_train(self, data_train):
        # if not self.K_train is None:
        #     print("Used a pre-trained Ktrain for MSK.")
        #     return self.K_train
        feature_vector = self.compute_feature_vector(data_train)
        K = np.dot(feature_vector, feature_vector.T).toarray()
        if self.normalize:
            K = self.normalize_train(K)
        self.K_train = K
        return K

    def compute_test(self, data_train, data_test):
        feature_vector_train = self.compute_feature_vector(data_train)
        feature_vector_test = self.compute_feature_vector(data_test)
        K = np.dot(feature_vector_test, feature_vector_train.T).toarray()
        if self.normalize:
            K = self.normalize_test(K, feature_vector_test)
        self.K_test = K
        return K

    def all_elements(self, i):
        if i == 1:
            return ["A", "C", "G", "T"]
        else:
            old_list = self.all_elements(i-1)
            new_list = []
            for e in old_list:
                new_list.extend([e + "A", e + "C", e + "G", e + "T"])
            return new_list

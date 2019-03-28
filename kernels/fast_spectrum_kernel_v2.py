import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from kernels.fast_kernel import FastKernel

class SpectrumKernel(FastKernel):
    def __init__(self, k, normalize=False):
        super().__init__()
        print("V2 with k={}".format(k))
        self.k = k
        self.index = {"A": 0, "T": 1, "C": 2, "G": 3}
        self.normalize = normalize

    def compute_index(self, word):
        res = 0
        for i in range(self.k):
            res += self.index[word[i]] * 4**i
        return res

    def compute_feature_vector(self, x):
        features =  lil_matrix((1, 4 ** self.k))
        for j in range(len(X) - self.k + 1):
            features[0, self.compute_index(list(line[j:j + self.k]))] += 1
        return csr_matrix(features)

    def compute_train(self, data_train):
        # feature_vector = self.compute_feature_vector(data_train)
        # K = np.dot(feature_vector, feature_vector.T).toarray()
        phi = self.phi_from_list(data_train)
        K = np.array(phi.dot(phi.T).todense())
        if self.normalize:
            K = self.normalize_train(K)
        return K

    def compute_test(self, data_train, data_test):
        # feature_vector_train = self.compute_feature_vector(data_train)
        # feature_vector_test = self.compute_feature_vector(data_test)
        # K = np.dot(feature_vector_test, feature_vector_train.T).toarray()
        phi_te = self.phi_from_list(data_test)
        phi_tr = self.phi_from_list(data_train)
        K = np.array(phi_te.dot(phi_tr.T).todense())
        if self.normalize:
            K = self.normalize_test(K, phi_te)
        return K

    def phi_from_list(self, X):
        m = len(X)
        l = len(X[0])
        X_encoded = np.zeros((m, l), dtype=np.uint8)
        for i in range(m):
            X_encoded[i] = [self.index[X[i][j]] for j in range(l)]

        T = np.zeros((l - self.k + 1, l), dtype=np.uint32)

        for i in range(self.k):
            diag = np.diagonal(T, i)
            diag.setflags(write=True)
            diag.fill(4** i)

        X_indexed = X_encoded.dot(T.T)
        Phi = lil_matrix((m, 4** self.k), dtype=np.float64)
        for i in range(X_indexed.shape[0]):
            tmp = np.bincount(X_indexed[i, :])
            Phi[i, :tmp.size] += tmp
        return Phi

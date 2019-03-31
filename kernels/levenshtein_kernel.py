########################################################################
### LinearKernel
########################################################################

from kernels.kernel import Kernel
import numpy as np
import Levenshtein

from tqdm import tqdm
import time

def levenshtein_distance(s1, s2):
    """ The Levenshtein distance between s1 and s2 """
    m = np.zeros((len(s1)+1, len(s2)+1), dtype=int)
    m[:, 0] = np.arange(0, len(s1)+1)
    m[0, :] = np.arange(0, len(s2)+1)
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                m[i, j] = np.min([m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1]])
            else:
                m[i, j] = np.min([m[i-1, j]+1, m[i, j-1]+1, \
                                        m[i-1, j-1]+1])
    return m[len(s1), len(s2)]

class LevenshteinKernel():
    """ LinearKernel class """

    def __init__(self, gamma, normalize = False):
        self.gamma = gamma
        self.normalize = normalize

    def evaluate(self, string1, string2):
        return np.exp(- self.gamma * Levenshtein.distance(string1, string2))

    def compute_train(self, Xtr):
        print("Computing Train Kernel ...")
        start = time.time()
        n = len(Xtr)
        K = np.zeros((n, n))
        pairs = []
        for i in range(n):
            for j in range(i + 1):
                pairs.append((i,j))
        for (i,j) in tqdm(pairs):
                K[j, i] = self.evaluate(Xtr[i], Xtr[j])

        # Symmetrize Kernel
        K = K + K.T - np.diag(K.diagonal())

        if self.normalize:
            K = self.normalize_train(K)

        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return K

    def compute_test(self, Xtr, Xte):
        print("Computing Test Kernel ...")
        start = time.time()
        n = len(Xtr)
        m = len(Xte)
        K_t = np.zeros((m, n))
        pairs = []
        for k in range(m):
            for j in range(n):
                pairs.append((k,j))
        for (k,j) in tqdm(pairs):
            K_t[k, j] = self.evaluate(Xte[k], Xtr[j])

        if self.normalize:
           K_t = self.normalize_test(K_t, Xte = Xte)

        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return K_t

    def normalize_train(self, K_train): #K_train unormalized
        self.norms_train = np.sqrt(K_train.diagonal())  # norms for x train vector
        matrix_norms = np.outer(self.norms_train,self.norms_train) #10e-40
        K_train =  np.divide(K_train, matrix_norms)
        return K_train

    def normalize_test(self, K_test, feats_test = None, Xte = None): #K_test unormalized
        m = K_test.shape[0]
        if feats_test is None:
            assert (not (Xte is None))
            norms_test = np.zeros(m)  # norms for x test vector
            for k in range(m):
                norms_test[k] = self.evaluate(Xte[k], Xte[k])
        else:
            assert (not (feats_test is None))
            norms_test = np.linalg.norm(feats_test,axis=1)
        matrix_norms = np.outer(norms_test,self.norms_train) #+ 1e-40  # matrix sqrt(K(xtest,xtest)*K(xtrain,xtrain))
        K_test = np.divide(K_test, matrix_norms)
        return K_test

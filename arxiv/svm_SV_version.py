import numpy as np
from cvxopt import matrix, solvers
import math

EPS = math.pow(10,-5)

class SVM_SV_version:
    #TODO : old version of SVM (with SV)
    """
        Implements Support Vector Machine.
    """

    def __init__(self, kernel=None, center=False):
        self.kernel = kernel

    def init_train(self, Xtr, Ytr, K):

        self.Xtr = Xtr
        self.Ytr = Ytr
        self.n = len(Xtr)

        if K is None:
            print("Building the Kernel ...")
            self.K = self.kernel.compute_train(self.Xtr)
            print("Kernel built successfully !")
        else:
            self.K = K

    def train(self, Xtr, Ytr, lambd=1, K=None):
        self.init_train(Xtr, Ytr, K)
        print("Solving SVM optimization, please wait ...")
        P = matrix(self.K, tc='d')
        q = matrix(-Ytr, tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)), np.diag(Ytr.astype(float)), axis=0), tc='d')
        h = matrix(np.append(np.zeros(self.n), np.ones(self.n, dtype=float) / (2 * lambd * self.n), axis=0), tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
        print("SVM optimization solved !")

        self.alpha = self.alpha.flatten()

        self.idx_SV = np.argwhere(np.abs(self.alpha)>EPS).flatten()
        print("number of SV : ", self.idx_SV.shape[0])
        self.Xtr_SV = {i:self.Xtr[self.idx_SV[i]] for i in range(self.idx_SV.shape[0])}
        self.alpha_SV = self.alpha[np.ix_(self.idx_SV)]
        print("alpha values of SV ", self.alpha_SV)

    def score_train(self):
        K_t = self.K[np.ix_(np.arange(self.n),self.idx_SV)]
        f = np.sign(K_t.dot(self.alpha_SV.reshape((self.alpha_SV.size, 1))))
        return np.mean(f.reshape(-1)==self.Ytr)

    def predict(self, Xte, K_t=None):
        print("Predicting Test sets, please wait ...")
        if K_t is None:
            self.K_t = self.kernel.compute_test(self.Xtr_SV, Xte)
        else:
            self.K_t = K_t
        Yte = self.K_t.dot(self.alpha_SV.reshape((self.alpha_SV.size, 1))).reshape(-1)
        print("Test predictions computed successfully !")
        return np.sign(Yte)

    def score(self, Xt, Yt):
        predictions = self.predict(Xt, K_t=None)
        #print(predictions,Yt)
        return np.mean(predictions==Yt)

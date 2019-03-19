
import numpy as np
from cvxopt import matrix, solvers
from kernels.centered_kernel import *

class SVM:
    """
        Implements Support Vector Machine.
    """

    def __init__(self, kernel=None, center=False):
        self.kernel = kernel
        self.center = center

    def init_train(self, Xtr, Ytr, K):

        self.Xtr = Xtr
        self.Ytr = Ytr
        self.n = len(Xtr)

        if self.center:
            print("Centered K")
            if not isinstance(self.kernel, CenteredKernel):
                self.kernel = CenteredKernel(self.kernel)
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


    def get_training_results(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return f.reshape(-1)

    def predict(self, Xte, K_t=None):
        print("Predicting Test sets, please wait ...")
        if K_t is None:
            self.K_t = self.kernel.compute_test(self.Xtr, Xte)
        else:
            self.K_t = K_t
        Yte = self.K_t.dot(self.alpha.reshape((self.alpha.size, 1))).reshape(-1)
        print("Test predictions computed successfully !")

        return Yte

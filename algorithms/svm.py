import numpy as np
from cvxopt import matrix, solvers
from kernels.centered_kernel import *
import math

EPS = math.pow(10,-5)

from cvxopt import matrix, solvers


class SVM:
    """
        Implements Support Vector Machine.
    """

    def __init__(self, kernel=None, center=False):
        self.kernel = kernel
        self.center = center

    def init_train(self, Xtr, Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.K = self.kernel.compute_train(self.Xtr)

    def train(self, Xtr, Ytr, lbd=1):
        n = len(Xtr)
        self.init_train(Xtr, Ytr)

        print("Solving SVM optimization ...")

        P = matrix(self.K, tc='d')
        q = matrix(-Ytr, tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)), np.diag(Ytr.astype(float)), axis=0), tc='d')
        h = matrix(np.append(np.zeros(n), np.ones(n, dtype=float) / (2 * lbd * n), axis=0), tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
        self.alpha[np.abs(self.alpha) < EPS] = 0

        print("SVM solved !")

    def get_training_results(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return f.reshape(-1)

    def predict(self, Xte):
        print("Predicting ...")
        self.K_t = self.kernel.compute_test(self.Xtr, Xte)
        predictions = self.K_t.dot(self.alpha.reshape((self.alpha.size, 1))).reshape(-1)
        print("End of predictions !")

        return predictions

    def score_train(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return np.mean(f.reshape(-1) == self.Ytr)

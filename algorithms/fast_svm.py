import numpy as np
from cvxopt import matrix, solvers
import math

EPS = math.pow(10,-5)

from sklearn import svm

class FastSVM:
    """
        Implements Support Vector Machine using sklearn.
        This class is only for parameter tuning.
    """

    def __init__(self, kernel=None, center=False, train_filename = None):
        self.kernel = kernel
        self.center = center


    def train(self, Xtr, Ytr, lbd=1, verbose=True):
        n = len(Xtr)
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.svm = svm.SVC(C=1./lbd, kernel='precomputed')

        try:
            self.kernel.load_kernel(train_filename)
            self.K = self.kernel.K
        except:
            if verbose:
                print("Computing train kernel ...")
            self.K = self.kernel.compute_train(self.Xtr)

        self.svm.fit(self.K, Ytr)
        support = self.svm.support_
        dc = self.svm.dual_coef_
        self.alpha = np.zeros(Xtr.shape[0])
        for idx, s in enumerate(support):
            self.alpha[s] = dc[0,idx]
        self.alpha = self.alpha.reshape(-1,1)

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

    def get_objective(self, Ytr):
        return -0.5 * self.alpha.T.dot(self.K).dot(self.alpha)[0, 0] + np.sum(self.alpha.flatten()*Ytr)
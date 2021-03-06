from imports import *
from cvxopt import matrix, solvers

EPS = math.pow(10,-5)

class SVM:
    """ Implements Support Vector Machine """

    def __init__(self, kernel=None, center=False, train_filename = None):
        self.kernel = kernel
        self.center = center
        self.train_filename = train_filename

    def train(self, Xtr, Ytr, lbd=1, verbose=True):
        n = len(Xtr)
        self.Xtr = Xtr
        self.Ytr = Ytr
        try:
            self.kernel.load_kernel(self.train_filename)
            self.K = self.kernel.K
        except:
            if verbose:
                print("Computing Training Gram Matrix ...")
            self.K = self.kernel.compute_train(self.Xtr)

        if verbose:
            print("Solving SVM optimization ...")

        P = matrix(self.K, tc='d')
        q = matrix(-Ytr, tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)), np.diag(Ytr.astype(float)), axis=0), tc='d')
        h = matrix(np.append(np.zeros(n), np.ones(n, dtype=float) / (2 * lbd * n), axis=0), tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
        self.alpha[np.abs(self.alpha) < EPS] = 0
        if verbose:
            print("SVM solved !")

    def get_training_results(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return f.reshape(-1)

    def predict(self, Xte):
        print("Predicting ...")
        self.K_t = self.kernel.compute_test(self.Xtr, Xte)
        predictions = self.K_t.dot(self.alpha.reshape((self.alpha.size, 1))).reshape(-1)
        print("End of predictions!")
        return predictions

    def score_train(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return np.mean(f.reshape(-1) == self.Ytr)

    def get_objective(self, Ytr):
        return -0.5 * self.alpha.T.dot(self.K).dot(self.alpha)[0, 0] + np.sum(np.abs(self.alpha.flatten()))

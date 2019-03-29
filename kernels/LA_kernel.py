from tqdm import tqdm
import numpy as np

import time
from scipy.linalg import sqrtm
from scipy.sparse.linalg import norm


class LAKernel():

    # http://cazencott.info/dotclear/public/publications/Azencott_KernelsForSequences09.pdf
    # https://www.cs.princeton.edu/~bee/courses/read/saigo-bioinformatics-2004.pdf?fbclid=IwAR0LjydQ9kD6FY8ISBsRiYGk_7iopyQAwVKGvzCFoF9EO-O6498av6NcRK4

    def __init__(self, nb_anchors_Nystrom=100, normalize=True):

        # nb anchors
        self.nb_anchors = nb_anchors_Nystrom
        print("Nb of anchor points (Nystrom) : ", self.nb_anchors)

        self.normalize = normalize

        # gap penalty
        self.d = 1.  # gap opening
        self.e = 1.  # 11. #extension cost

        # against diagonal dominance
        self.beta = 0.5

        # csts used in dynamic programming
        self.cst1 = np.exp(self.beta * self.d)
        self.cst2 = np.exp(self.beta * self.e)

        self.type = "BLAST"  # or "Transversion" #or ""Identity"

    def score_substitution(self, x, y):
        # substitution table
        # https://en.wikipedia.org/wiki/Substitution_matrix
        # (https://en.wikipedia.org/wiki/Models_of_DNA_evolution)
        # https://slideplayer.com/slide/5092656/

        if self.type == "BLAST":
            equal = int(x == y)
            return 5 * equal - 4 * (1 - equal)
        elif self.type == "Transversion":
            equal = int(x == y)
            return 1 * equal - 5 * (1 - equal)
        elif self.type == "Identity":
            equal = int(x == y)
            return equal
        else:
            exit()

    def evaluate(self, x, y):
        """
        Evaluation function computing the inner product between phi_x and phi_y
        """

        nx = len(x) + 1
        ny = len(y) + 1
        M = np.zeros((nx, ny))
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        X2 = np.zeros((nx, ny))
        Y2 = np.zeros((nx, ny))

        for i in range(1, nx):
            for j in range(1, ny):
                # print(cst1,cst2)
                s = self.score_substitution(x[i - 1], y[j - 1])
                M[i, j] = np.exp(self.beta * s) * (1 + X[i - 1][j - 1] + Y[i - 1][j - 1] + M[i - 1][j - 1])
                X[i, j] = self.cst1 * M[i - 1, j] + self.cst2 * X[i - 1, j]
                Y[i, j] = self.cst1 * (M[i, j - 1] + X[i, j - 1]) + self.cst2 * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]

        # print(1 + X2[-1,-1] + Y2[-1,-1] + M[-1,-1])

        return 1 / self.beta * np.log(1 + X2[-1, -1] + Y2[-1, -1] + M[-1, -1])

    def choose_anchor_points(self, Xtr, option="random"):

        n = len(Xtr)
        if option=="random":
            self.anchors = np.random.choice(np.arange(n), self.nb_anchors)
        else:
            print("Only random selection of anchor points was implemented")
            print("Performs random selection of anchor points")
            self.anchors = np.random.choice(np.arange(n), self.nb_anchors)

    def compute_K_anchors_Nystrom(self, Xtr):

        print("Computing K anchors (Nystrom) ...")
        start = time.time()
        n = len(Xtr)

        self.idx_anchors = {self.anchors[k]: k for k in range(self.nb_anchors)}

        self.K_anchors = np.zeros((self.nb_anchors, self.nb_anchors))
        pairs = []
        for i in range(self.nb_anchors):
            for j in range(i + 1):
                pairs.append((self.anchors[i], self.anchors[j]))

        for (i, j) in tqdm(pairs):
            self.K_anchors[self.idx_anchors[j], self.idx_anchors[i]] = self.evaluate(Xtr[i], Xtr[j])

        # Symmetrize Kernel
        self.K_anchors = self.K_anchors + self.K_anchors.T - np.diag(self.K_anchors.diagonal())

        eigenv = None  # smallest negative eigenvalue
        for e in np.real(np.linalg.eigvals(self.K_anchors)):
            if e < 0:
                if eigenv is None:
                    eigenv = e
                else:
                    eigenv = max(eigenv, e)
        if not (eigenv is None):
            self.K_anchors = self.K_anchors - eigenv * np.eye(self.nb_anchors)

        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))

        self.Xtr = Xtr
        self.inv_sqrt_K_anchors = sqrtm(np.linalg.inv(self.K_anchors)) #square root of matrix
        #print(self.inv_sqrt_K_anchors)
        self.inv_sqrt_K_anchors = np.real(self.inv_sqrt_K_anchors) #discard infinitesimal imaginary part
        # print("inv sqrt anchors : ", self.inv_sqrt_K_anchors.shape)

        self.is_anchor = np.zeros(n)
        for i in range(n):
            if i in self.anchors.tolist():
                self.is_anchor[i] = 1

    def compute_features_train_Nystrom(self):

        print("Computing train features (Nystrom) ...")

        n = len(self.Xtr)

        eval_with_anchors =  np.zeros((n,self.nb_anchors))
        pairs = []
        for i in range(n):
            if self.is_anchor[i]:
                eval_with_anchors[i, :] = self.K_anchors[self.idx_anchors[i]]
            else:
                for k in range(self.nb_anchors):
                    pairs.append((i,self.anchors[k]))

        for (i,j) in tqdm(pairs):
            eval_with_anchors[i, self.idx_anchors[j]] =  self.evaluate(self.Xtr[i], self.Xtr[j])

        #print("eval train with anchors shape : ", eval_with_anchors.shape)

        self.features_train = self.inv_sqrt_K_anchors.dot(eval_with_anchors.T).T

        print("features train shape : ",self.features_train.shape)

    def compute_train(self, Xtr):

        print("Nb of training points : ", Xtr.shape[0])

        self.choose_anchor_points(Xtr)
        self.compute_K_anchors_Nystrom(Xtr)
        self.compute_features_train_Nystrom()

        K_train = np.dot(self.features_train, self.features_train.T)

        if self.normalize:
            K_train = self.normalize_train(K_train)

        print("Ktrain shape ", K_train.shape)
        #print("Ktrain",K_train)

        return K_train

    def compute_test(self, Xtr, Xte): #Xtr not used. We use self.Xtr

        print("Computing test features (Nystrom) ...")

        n = len(Xte)

        eval_with_anchors = np.zeros((n, self.nb_anchors))
        pairs = []
        for i in range(n):
            for k in range(self.nb_anchors):
                pairs.append((i, self.anchors[k]))

        for (i, j) in tqdm(pairs):
            eval_with_anchors[i, self.idx_anchors[j]] = self.evaluate(Xte[i], self.Xtr[j])

        features_test = self.inv_sqrt_K_anchors.dot(eval_with_anchors.T).T
        print("features test shape : ", features_test.shape)

        K_test = np.dot(features_test, self.features_train.T)

        if self.normalize:
            K_test = self.normalize_test(K_test, features_test)

        print("Ktest shape ", K_test.shape)
        #print("Ktest ", K_test)

        return K_test

    def normalize_train(self, K_train): #K_train unormalized
        self.norms_train = np.sqrt(K_train.diagonal())  # norms for x train vector
        matrix_norms = np.outer(self.norms_train,self.norms_train) #10e-40
        K_train =  np.divide(K_train, matrix_norms)
        return K_train

    def normalize_test(self, K_test, feats_test): #K_test unormalized
        norms_test = np.linalg.norm(feats_test,axis=1)
        matrix_norms = np.outer(norms_test,self.norms_train) #+ 1e-40  # matrix sqrt(K(xtest,xtest)*K(xtrain,xtrain))
        K_test = np.divide(K_test, matrix_norms)
        return K_test

    '''
    def compute_train(self, Xtr):
        print("Computing Train Kernel ...")
        start = time.time()
        n = len(Xtr)
        K = np.zeros((n, n))
        pairs = []
        for i in range(n):
            for j in range(i + 1):
                pairs.append((i, j))
        for (i, j) in tqdm(pairs):
            K[j, i] = self.evaluate(Xtr[i], Xtr[j])
        # Symmetrize Kernel
        K = K + K.T - np.diag(K.diagonal())

        eigenv = None  # smallest negative eigenvalue
        for e in np.real(np.linalg.eigvals(K)):
            if e < 0:
                if eigenv is None:
                    eigenv = e
                else:
                    eigenv = max(eigenv, e)
        if not (eigenv is None):
            K = K - eigenv * np.eye(n)

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
                pairs.append((k, j))

        for (k, j) in tqdm(pairs):
            K_t[k, j] = self.evaluate(Xte[k], Xtr[j])

        # if not(self.smallest_neg_eigenv_train is None):
        #    K_t = K_t - self.smallest_neg_eigenv_train * np.eye(m)[:,:n]

        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return K_t
    '''

















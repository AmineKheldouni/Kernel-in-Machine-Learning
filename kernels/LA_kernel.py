########################################################################
### LocalAlignmentKernel
########################################################################

from imports import *

from scipy.linalg import sqrtm
from scipy.sparse.linalg import norm

class LAKernel():

    """Our implementation of the Local Alignment Kernel, with Nystrom approximation (for speed up)"""
    #
    def __init__(self, nb_anchors_Nystrom=100, gap_costs = (1,7), type="BLAST", beta=0.5, normalize=False):

        # nb anchors
        self.nb_anchors = nb_anchors_Nystrom
        print("Nb of anchor points (Nystrom) : ", self.nb_anchors)

        self.normalize = normalize

        # gap penalty
        self.d = gap_costs[0]  # gap opening
        self.e = gap_costs[1]  #extension cost

        # against Gram matrix diagonal dominance
        self.beta = beta

        # csts used in dynamic programming
        self.cst1 = np.exp(self.beta * self.d)
        self.cst2 = np.exp(self.beta * self.e)

        self.type = type  # "BLAST" or "Transversion" or ""Identity"

    def score_substitution(self, x, y):
        # substitution table. Source : https://slideplayer.com/slide/5092656/

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
        #Evaluation function computing the K(x,y)

        nx = len(x) + 1
        ny = len(y) + 1
        M = np.zeros((nx, ny))
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        X2 = np.zeros((nx, ny))
        Y2 = np.zeros((nx, ny))

        for i in range(1, nx):
            for j in range(1, ny):
                s = self.score_substitution(x[i - 1], y[j - 1])
                M[i, j] = np.exp(self.beta * s) * (1 + X[i - 1][j - 1] + Y[i - 1][j - 1] + M[i - 1][j - 1])
                X[i, j] = self.cst1 * M[i - 1, j] + self.cst2 * X[i - 1, j]
                Y[i, j] = self.cst1 * (M[i, j - 1] + X[i, j - 1]) + self.cst2 * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]

        return 1 / self.beta * np.log(1 + X2[-1, -1] + Y2[-1, -1] + M[-1, -1])

    def choose_anchor_points(self, Xtr, option="random"):
        # Choose self.nb_anchors anchor points among training set for Nystrom approximation

        n = len(Xtr)
        if option=="random":
            self.anchors = np.sort(np.random.choice(np.arange(n), self.nb_anchors, replace = False))
        else:
            print("Only random selection of anchor points was implemented")
            print("Performs random selection of anchor points")
            self.anchors = np.sort(np.random.choice(np.arange(n), self.nb_anchors, replace = False))

        self.is_anchor = np.zeros(n)
        for i in range(n):
            if i in self.anchors.tolist():
                self.is_anchor[i] = 1

    def compute_K_anchors_Nystrom(self, Xtr):
        # Compute the exact Gram matrix for the anchor points

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
                    eigenv = min(eigenv, e)
        if not (eigenv is None):
            self.K_anchors = self.K_anchors - (eigenv - 1) * np.eye(self.nb_anchors) #to make it positive definite

        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))

        self.Xtr = Xtr
        self.inv_sqrt_K_anchors = sqrtm(np.linalg.inv(self.K_anchors)) #square root of matrix
        self.inv_sqrt_K_anchors = np.real(self.inv_sqrt_K_anchors)  #not necessary actually (discard infinitesimal imaginary part)
        # print("inv sqrt anchors : ", self.inv_sqrt_K_anchors.shape)

    def compute_features_train_Nystrom(self):
        # Compute the Nystrom features for the train data

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
            eval_with_anchors[i, self.idx_anchors[j]] = self.evaluate(self.Xtr[i], self.Xtr[j])
        #print("eval train with anchors shape : ", eval_with_anchors.shape)

        self.features_train = self.inv_sqrt_K_anchors.dot(eval_with_anchors.T).T
        print("features train shape : ",self.features_train.shape)

    def compute_train(self, Xtr):
        # Compute the approximate training Gram matrix (using Nystrom features)

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

    def compute_test(self, Xtr, Xte):
        # Compute the approximate test Gram matrix (using Nystrom features)

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

    def normalize_train(self, K_train): #to normalize Ktrain
        self.norms_train = np.sqrt(K_train.diagonal())  # norms for x train vector
        matrix_norms = np.outer(self.norms_train,self.norms_train) #10e-40
        K_train =  np.divide(K_train, matrix_norms)
        return K_train

    def normalize_test(self, K_test, feats_test): #to normalize Ktest
        norms_test = np.linalg.norm(feats_test,axis=1)
        matrix_norms = np.outer(norms_test,self.norms_train) #+ 1e-40  # matrix sqrt(K(xtest,xtest)*K(xtrain,xtrain))
        K_test = np.divide(K_test, matrix_norms)
        return K_test














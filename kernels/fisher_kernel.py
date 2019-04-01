########################################################################
### Fisher Kernel (HMM modelling)
########################################################################

from imports import *
from kernels.kernel import Kernel


# parameters for dataset 0 for a simple model with markov assumption of 1 and 2 hidden states
# pi_0 = np.array([0.49157036, 0.50842964])
# pi_fin = np.array([0.49789493, 0.50210507])
# p = np.array([[0.28160794, 0.00459951, 0.52885727, 0.18493528],
#  [0.17609247, 0.54436512, 0.00826462, 0.27127779]])
# A = np.array([[0.61262946, 0.38737054],
#  [0.39016698, 0.60983302]])
# theta = [A, p, pi_0, pi_fin]

class FisherKernel(Kernel):
    """ LinearKernel class """

    def __init__(self, k, normalize):
        super().__init__()
        self.k = k
        permutations = self.generate_permutations(k)
        self.index = dict(zip(permutations, np.arange(len(permutations))))
        self.normalize = normalize

    def logplus(self, x, y):
        M = np.maximum(x, y)
        m = np.minimum(x, y)
        return M + np.log(1 + np.exp(m - M))

    def log_pdf(self, x, p):
        return np.log(p[self.index[x]])

    # u is only one sequence
    def alpha_recursion(self, u, A, pi_0, p):
        T = len(u)
        K = len(p)

        A = np.log(A)

        alpha = np.zeros((T, K), dtype=np.float64)

        alpha[0] = np.log(pi_0) + np.array([self.log_pdf(u[0], p[k]) for k in range(K)]).squeeze()

        for t in range(1, T):
            vec = np.array([
                self.log_pdf(u[t], p[k])
                for k in range(K)
            ]).squeeze()

            total = alpha[t - 1, 0] + A[0]
            for k in range(1, K):
                total = self.logplus(total, alpha[t - 1, k] + A[k])

            alpha[t] = vec + total

        return alpha

    def beta_recursion(self, u, A, pi_fin, p):
        T = len(u)
        K = len(p)

        A = np.log(A)

        beta = np.zeros((T, K))

        beta[T - 1] = np.log(pi_fin)

        for t in range(T - 2, -1, -1):
            vec = np.array([
                self.log_pdf(u[t + 1], p[k])
                for k in range(K)
            ]).squeeze()

            total = A[:, 0] + vec[0] + beta[t + 1, 0]
            for k in range(1, K):
                total = self.logplus(total, A[:, k] + vec[k] + beta[t + 1, k])

            beta[t] = total

        return beta

    def proba_hidden(self, t, alpha, beta):
        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return np.exp(prod - total)

    def proba_joint_hidden(self, t, u, alpha, beta, A, p):
        A = np.log(A)

        vec = np.array([
            self.log_pdf(u[t + 1], p[i])
            for i in range(len(p))
        ]).squeeze()
        matrix = alpha[t].reshape(-1, 1) + A + beta[t + 1].reshape(1, -1) + vec.reshape(1, -1)

        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return np.exp(matrix - total)

    def log_likelihood_hmm(self, t, alpha, beta):
        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return total

    def compute_feature_vector(self, X_initial, p, A, pi_0, pi_fin):
        X = self.transform(X_initial, self.k)
        T = len(X[0])
        K = len(p)
        indicator_matrix = self.make_matrix(X)
        alphas = []
        betas = []
        for string in X:
            alphas.append(self.alpha_recursion(string, A, pi_0, p))
            betas.append(self.beta_recursion(string, A, pi_fin, p))

        A2 = np.array([sum(self.proba_joint_hidden(t, string, alpha, beta, A, p) for t in range(T - 1))
                       for string, alpha, beta in zip(X, alphas, betas)])
        features = (A2 / A - A2.sum(axis=2).reshape(len(X), -1, 1)).reshape(len(X), -1)

        p_zt = np.array([[self.proba_hidden(t, alpha, beta) for t in range(T)] for alpha, beta in zip(alphas, betas)])

        p2 = np.zeros((len(X), p.shape[0], p.shape[1]))
        for t in range(len(X)):
            for k in range(K):
                for lettre in range(p.shape[1]):
                    p2[t, k, lettre] = (p_zt[t, :, k] * indicator_matrix[lettre, t, :]).sum()

        features = np.concatenate((features, (p2 / p - p2.sum(axis=2).reshape(len(X), -1, 1)).reshape(len(X), -1)),
                                  axis=1)
        return features

    ### Functions to find the likeliest parameters
    def EM_HMM(self, X_initial, K, A, pi_0, pi_fin, p, Niter=10):
        X = self.transform(X_initial, self.k)

        np.random.seed(10)
        T = len(X[0])
        l = len(X)
        indicator_matrix = self.make_matrix(X)

        loss = []

        for n in range(Niter):
            print(n)
            alphas = []
            betas = []
            for string in X:
                alphas.append(self.alpha_recursion(string, A, pi_0, p))
                betas.append(self.beta_recursion(string, A, pi_fin, p))

            p_zt = np.array(
                [[self.proba_hidden(t, alpha, beta) for t in range(T)] for alpha, beta in zip(alphas, betas)])
            pi_0 = p_zt.sum(axis=0)[0]
            pi_0 /= pi_0.sum()

            pi_fin = p_zt.sum(axis=0)[-1]
            pi_fin /= pi_fin.sum()

            A = sum(self.proba_joint_hidden(t, string, alpha, beta, A, p) for t in range(T - 1)
                    for string, alpha, beta in
                    zip(X, alphas, betas))
            A /= A.sum(axis=1).reshape(-1, 1)

            for k in range(K):
                for lettre in range(p.shape[1]):
                    p[k, lettre] = (p_zt[:, :, k] * indicator_matrix[lettre, :, :]).sum()
            p /= p.sum(axis=1).reshape(-1, 1)

            loss.append(sum(self.log_likelihood_hmm(0, alpha, beta) for alpha, beta in zip(alphas, betas)))

        return A, pi_0, pi_fin, p, loss

    def make_matrix(self, X):
        X_bis = []
        for string in X:
            X_bis.append(list(string))
        X_bis = np.array(X_bis)
        permutations = self.generate_permutations(self.k)
        return np.array([X_bis == perm for perm in permutations])  # .reshape(len(X), len(X[0]), -1, 4)

    def transform(self, X, k):
        X2 = []
        for x in X:
            temp = []
            for i in range(len(x) - k):
                temp.append(x[i:i + k])
            X2.append(temp)
        return X2

    def generate_permutations(self, k):
        if k == 1:
            return ["A", "C", "G", "T"]
        else:
            l = self.generate_permutations(k - 1)
            l2 = []
            for e in l:
                l2.append(e + "A")
                l2.append(e + "C")
                l2.append(e + "G")
                l2.append(e + "T")

            return l2


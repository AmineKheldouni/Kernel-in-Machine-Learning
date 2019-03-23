import numpy as np
from kernels.fast_kernel import FastKernel

# parameters for dataset 0
pi_0 = np.array([0.49157036, 0.50842964])
pi_fin = np.array([0.49789493, 0.50210507])
p = np.array([[0.28160794, 0.00459951, 0.52885727, 0.18493528],
 [0.17609247, 0.54436512, 0.00826462, 0.27127779]])
A = np.array([[0.61262946, 0.38737054],
 [0.39016698, 0.60983302]])
theta = [A, p, pi_0, pi_fin]

class FisherKernel(FastKernel):
    """ LinearKernel class """

    def __init__(self):
        self.index = {"A": 0, "C": 1, "G": 2, "T": 3}

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

    def compute_thetas(self, theta, eps):
        A, p, pi_0, pi_fin = theta
        thetas = []

        for i in range(len(A)):
            for j in range(len(A[i])):
                A2 = A.copy()
                A2[i, j] += eps
                A2 /= A2.sum(axis=1).reshape(-1, 1)
                thetas.append([A2, p, pi_0, pi_fin])

        for i in range(len(p)):
            for j in range(len(p[i])):
                p2 = p.copy()
                p2[i, j] += eps
                p2 /= p2.sum(axis=1).reshape(-1, 1)
                thetas.append([A, p2, pi_0, pi_fin])

        for i in range(len(pi_0)):
            pi_02 = pi_0.copy()
            pi_02[i] += eps
            pi_02 /= pi_02.sum()
            thetas.append([A, p, pi_02, pi_fin])

        for i in range(len(pi_fin)):
            pi_fin2 = pi_fin.copy()
            pi_fin2[i] += eps
            pi_fin2 /= pi_fin2.sum()
            thetas.append([A, p, pi_0, pi_fin2])

        return thetas

    def compute_likelihoods(self, X, thetas):
        likelihoods = []
        for theta in thetas:
            A, p, pi_0, pi_fin = theta

            alphas = []
            betas = []
            for string in X:
                alphas.append(self.alpha_recursion(string, A, pi_0, p))
                betas.append(self.beta_recursion(string, A, pi_fin, p))

            likelihoods.append([self.log_likelihood_hmm(0, alpha, beta) for alpha, beta in zip(alphas, betas)])
        return np.array(likelihoods)

    def compute_feature_vector(self, X, theta = theta, eps = 0.005):
        thetas = self.compute_thetas(theta, eps)
        initial_likelihood = self.compute_likelihoods(X, [theta])
        likelihoods = self.compute_likelihoods(X, thetas)
        return (initial_likelihood - likelihoods).T / eps

    ### Functions to fing the likeliest parameters

    def EM_HMM(self, X, K, A, pi_0, pi_fin, p, Niter=10):
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

            p_zt = np.array([[self.proba_hidden(t, alpha, beta) for t in range(T)] for alpha, beta in zip(alphas, betas)])
            pi_0 = p_zt.sum(axis=0)[0]
            pi_0 /= pi_0.sum()

            pi_fin = p_zt.sum(axis=0)[-1]
            pi_fin /= pi_fin.sum()

            A = sum(self.proba_joint_hidden(t, string, alpha, beta, A, p) for t in range(T - 1) for string, alpha, beta in
                    zip(X, alphas, betas))
            A /= A.sum(axis=1).reshape(-1, 1)

            for k in range(K):
                for lettre in range(p.shape[1]):
                    p[k, lettre] = (p_zt[:, :, k] * indicator_matrix[lettre, :, :]).sum()
            p /= p.sum(axis=1).reshape(-1, 1)

            loss.append(sum(self.log_likelihood_hmm(0, alpha, beta) for alpha, beta in zip(alphas, betas)))
            print(A, p, pi_0, pi_fin, loss[-1])

        return A, pi_0, pi_fin, p, loss

    def make_matrix(self, X):
        X_bis = []
        for string in X:
            X_bis.append(list(string))
        X_bis = np.array(X_bis)
        return np.array([X_bis == "A", X_bis == "C", X_bis == "G", X_bis == "T"])  # .reshape(len(X), len(X[0]), -1, 4)



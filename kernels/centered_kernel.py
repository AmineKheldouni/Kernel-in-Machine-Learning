########################################################################
################### CenteredKernel #####################################
########################################################################

from kernels.kernel import *

class CenteredKernel(Kernel):

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.eta = None

    def evaluate(self, x, y):
        tmp = 0.0
        for k in range(self.n):
            tmp += self.kernel.evaluate(x, self.Xtr[k])
            tmp += self.kernel.evaluate(y, self.Xtr[k])

        return self.kernel.evaluate(x, y) - tmp / self.n + self.eta


    def compute_K_train(self, Xtr, n, K=None):

        start = time.time()
        print("Called compute_K_train (CENTERED VERSION)")

        self.Xtr = Xtr
        self.n = n
        if K is None:
            # Compute the non-centered Gram matrix
            self.K_tr_nc = self.kernel.compute_K_train(self.Xtr, self.n)
        else:
            self.K_tr_nc = K

        self.eta = np.sum(self.K_tr_nc) / np.power(self.n, 2)

        # Store centered kernel
        U = np.ones(self.K_tr_nc.shape) / self.n
        self.K_tr_c = (np.eye(self.n) - U).dot(self.K_tr_nc) \
            .dot(np.eye(self.n) - U)

        end = time.time()
        print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return self.K_tr_c

    def compute_K_test(self, Xtr, n, Xte, m):
        print("Called compute_K_test (CENTERED VERSION)")
        start = time.time()

        if self.eta is None:
            self.compute_K_train(Xtr, n)

        # Get the non-centered K test
        K_te_nc = self.kernel.compute_K_test(Xtr, n, Xte, m)

        # The new K_t is the non-centered matrix
        K_te_c = K_te_nc + (-1 / self.n) * (
                    K_te_nc.dot(np.ones((self.n, self.n))) +
                    np.ones((m, self.n)).dot(self.K_tr_nc))
        K_te_c += self.eta

        end = time.time()
        print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_te_c

    def get_non_centered_K_tr(self):
        return self.K_tr_nc

    def get_centered_K_tr(self):
        return self.K_tr_c

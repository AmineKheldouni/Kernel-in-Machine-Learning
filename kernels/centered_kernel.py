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

    def compute_train(self, Xtr, K=None):
        start = time.time()
        print("Compute Train: Centered Kernel")
        self.Xtr = Xtr
        self.n = len(self.Xtr)
        if K is None:
            # Compute the non-centered Gram matrix
            self.K = self.kernel.compute_train(self.Xtr)
        else:
            self.K = K

        self.eta = np.sum(self.K) / np.power(self.n, 2)

        # Compute centered kernel
        U = (1./self.n) * np.ones(self.K.shape)
        self.K_centered = (np.eye(self.n) - U).dot(self.K).dot(np.eye(self.n) - U)
        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return self.K_centered

    def compute_test(self, Xtr, Xte):
        print("Compute Test: Centered Kernel")
        start = time.time()
        n = len(Xtr)
        m = len(Xte)

        if self.eta is None:
            self.compute_train(Xtr)

        # Get the non-centered Test Kernel
        K = self.kernel.compute_test(Xtr, Xte)
        # Compute the centered version
        Kte_centered = K + (-1 / self.n) * (K.dot(np.ones((self.n, self.n))) + np.ones((m, self.n)).dot(self.K))
        Kte_centered += self.eta
        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return Kte_centered

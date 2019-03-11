########################################################################
### Kernel
########################################################################

from abc import abstractmethod, ABC
import time
import numpy as np

class Kernel(ABC):

    def __init__(self):
        return

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def _fill_train_line(self, Xtr, i):
        res = np.zeros((i + 1,))
        for j in range(i + 1):
            res[j] = self.evaluate(Xtr[i], Xtr[j])
        return res

    def _fill_test_column(self, Xte, Xtr, j, m):
        res = np.zeros((m,))
        for k in range(m):
            res[k] = self.evaluate(Xte[k], Xtr[j])
        return res

    def compute_K_train(self, Xtr, n):
        print("Called compute_K_train")

        start = time.time()

        K = np.zeros([n, n], dtype=np.float64)

        for i in range(n):
            K[:i + 1, i] = self._fill_train_line(Xtr, i)

        # Symmetrize kernel
        K = K + K.T - np.diag(K.diagonal())

        end = time.time()
        print("Time elapsed:", "{0:.2f}".format(end - start))
        return K

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):

        print("Called compute_K_test (NON CENTERED VERSION)")
        start = time.time()

        K_t = np.zeros((m, n))

        for j in range(n):
            K_t[:, j] = self._fill_test_column(Xte, Xtr, j, m)

        if verbose:
            end = time.time()
            print("Time elapsed:", "{0:.2f}".format(end - start))

        return K_t

########################################################################
### Kernel
########################################################################

from abc import abstractmethod, ABC
import time
import numpy as np
from tqdm import tqdm

class Kernel(ABC):

    def __init__(self):
        return

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def compute_train(self, Xtr):
        print("Computing Train Kernel ...")
        start = time.time()
        n = len(Xtr)
        K = np.zeros((n, n))
        pairs = []
        for i in range(n):
            for j in range(i + 1):
                pairs.append((i,j))
        for (i,j) in tqdm(pairs):
                K[j, i] = self.evaluate(Xtr[i], Xtr[j])
        # Symmetrize Kernel
        K = K + K.T - np.diag(K.diagonal())
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
                pairs.append((k,j))
        for (k,j) in tqdm(pairs):
            K_t[k, j] = self.evaluate(Xte[k], Xtr[j])
        end = time.time()
        print("Time elapsed: {0:.2f}".format(end - start))
        return K_t

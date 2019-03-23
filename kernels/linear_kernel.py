########################################################################
### LinearKernel
########################################################################

from kernels.kernel import *
import numpy as np

class LinearKernel(Kernel):
    """ LinearKernel class """

    def __init__(self, normalize=True):
        super().__init__(normalize)

    def evaluate(self, x, y):
        return x.dot(y.transpose())

    def compute_train(self, Xtr):
        print("Compute Train: Linear Kernel")
        Ktr = Xtr.dot(Xtr.T)

        if self.normalize:
            D = np.diag(Ktr.diagonal())
            normalization_term = np.sqrt(D.dot(D.T))
            Ktr = np.divide(Ktr, normalization_term)

        print("end")
        return Ktr

    def compute_test(self, Xtr, Xte):
        print("Compute Test: Linear Kernel")
        K_te = Xte.dot(Xtr.T)

        if self.normalize:
            D = np.diag(K_te.diagonal())
            normalization_term = np.sqrt(D.dot(D.T))
            K_te = np.divide(K_te, normalization_term)
            
        print("end")
        return K_te

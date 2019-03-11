########################################################################
### LinearKernel
########################################################################

from kernels.kernel import *
import numpy as np

class LinearKernel(Kernel):
    """ LinearKernel class """

    def __init__(self):
        super().__init__()

    def evaluate(self, x, y):
        return x.dot(y.transpose())

    def compute_K_train(self, Xtr, n):

        print("LinearKernel.compute_K_train")
        Ktr = Xtr.dot(Xtr.T)
        print("end")

        return Ktr

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):

        print("LinearKernel.compute_K_test")
        K_te = Xte.dot(Xtr.T)
        print("end")

        return K_te

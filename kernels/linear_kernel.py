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

    def compute_train(self, Xtr):
        print("Compute Train: Linear Kernel")
        Ktr = Xtr.dot(Xtr.T)
        print("end")
        return Ktr

    def compute_test(self, Xtr, Xte):
        print("Compute Test: Linear Kernel")
        K_te = Xte.dot(Xtr.T)
        print("end")
        return K_te

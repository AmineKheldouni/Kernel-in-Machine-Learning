from kernels.fast_kernel import FastKernel
import numpy as np
from kernels.fast_spectrum_kernel import SpectrumKernel
import scipy.sparse as ss

class SumKernel():
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels

    def compute_train(self, data_train):
        K = 0
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_train(data_train)
        return K

    def compute_test(self, data_train, data_test):
        K = 0
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_test(data_train, data_test)
            #give the option to normalize each kernel
        return K

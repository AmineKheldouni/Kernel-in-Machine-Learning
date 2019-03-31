from kernels.kernel import Kernel
import numpy as np
from kernels.spectrum_kernel import SpectrumKernel
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
import scipy.sparse as ss
import glob

class SumKernel():
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels

    def compute_train(self, data_train):
        K = 0
        print("Compute K train ...")
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_train(data_train)
        self.K = K
        return K

    def compute_test(self, data_train, data_test):
        K = 0
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_test(data_train, data_test)
            #give the option to normalize each kernel
        return K

    def load(self, dataset_idx):
        """ Function to load pre-trained train kernel matrices for
        MismatchSpectrumKernels
        """
        self.kernels = []
        for f in glob.glob("./storage/{}/*.npy".format(dataset_idx)):
            Kt = np.load(f)
            params = f.split(',')
            msk = MismatchSpectrumKernel(int(params[0][-1]),
                                       int(params[1][0]),
                                       normalize=True)
            msk.K_train = Kt
            self.kernels.append(msk)

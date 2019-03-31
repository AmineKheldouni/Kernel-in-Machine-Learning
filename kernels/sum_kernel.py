########################################################################
### Sum of Kernels
########################################################################

from imports import *
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
import glob

class SumKernel():
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels

    def compute_train(self, data_train):
        K = 0
        print("Computing Training Gram Matrix ...")
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_train(data_train)
        self.K = K
        return K

    def compute_test(self, data_train, data_test):
        K = 0
        print("Computing Test Gram Matrix ...")
        for i in range(len(self.kernels)):
            K += self.kernels[i].compute_test(data_train, data_test)
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

########################################################################
### Weighted Sum of Kernels
########################################################################

from imports import *

class WeightedSumKernel():
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels
        self.nb_kernels = len(self.kernels)
        self.etas = 1/self.nb_kernels * np.ones(self.nb_kernels)

    def precompute_train(self, data_train):
        self.K_trains = []
        for i in range(len(self.kernels)):
            self.K_trains.append(self.kernels[i].compute_train(data_train))

    def compute_train(self, data_train):
        #data train not used but kept to make svm class compatible
        K = 0
        for i in range(len(self.kernels)):
            K += self.etas[i] * self.K_trains[i]
        return K

    def compute_test(self, data_train, data_test):
        K = 0
        for i in range(len(self.kernels)):
            K += self.etas[i] * self.kernels[i].compute_test(data_train, data_test)
        return K

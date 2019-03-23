########################################################################
### LinearKernel
########################################################################

from kernels.fast_kernel import FastKernel
import numpy as np

class LinearKernel(FastKernel):
    def __init__(self):
        super().__init__()

    def string_to_vec(self, string):
        vec = []
        for letter in string:
            if letter == "A":
                vec.extend([1, 0, 0, 0])
            elif letter == "C":
                vec.extend([0, 1, 0, 0])
            elif letter == "G":
                vec.extend([0, 0, 1, 0])
            elif letter == "T":
                vec.extend([0, 0, 0, 1])

        return vec

    def compute_feature_vector(self, data):
        feature_vector = []
        for string in data:
            feature_vector.append(self.string_to_vec(string))
        return np.array(feature_vector)

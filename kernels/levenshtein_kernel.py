########################################################################
### LinearKernel
########################################################################

from kernels.kernel import Kernel
import numpy as np
import Levenshtein

class LevenshteinKernel(Kernel):
    """ LinearKernel class """

    def __init__(self, gamma, normalize = False):
        super().__init__(normalize = normalize)
        self.gamma = gamma

    def evaluate(self, string1, string2):
        return np.exp(- self.gamma * Levenshtein.distance(string1, string2))

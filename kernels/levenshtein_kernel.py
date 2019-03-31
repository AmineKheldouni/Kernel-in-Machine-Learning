########################################################################
### LinearKernel
########################################################################

from kernels.kernel import Kernel
import numpy as np
import Levenshtein


def levenshtein_distance(s1, s2):
    """ The Levenshtein distance between s1 and s2 """
    m = np.zeros((len(s1)+1, len(s2)+1), dtype=int)
    m[:, 0] = np.arange(0, len(s1)+1)
    m[0, :] = np.arange(0, len(s2)+1)
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                m[i, j] = np.min([m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1]])
            else:
                m[i, j] = np.min([m[i-1, j]+1, m[i, j-1]+1, \
                                        m[i-1, j-1]+1])
    return m[len(s1), len(s2)]

class LevenshteinKernel(Kernel):
    """ LinearKernel class """

    def __init__(self, gamma, normalize = False):
        super().__init__(normalize = normalize)
        self.gamma = gamma

    def evaluate(self, string1, string2):
        return np.exp(- self.gamma * Levenshtein.distance(string1, string2))

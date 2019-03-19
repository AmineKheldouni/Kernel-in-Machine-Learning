########################################################################
### SpectrumKernel
########################################################################

from kernels.kernel import *

class SpectrumKernel(Kernel):
    """ SpectrumKernel class """

    def __init__(self, k, EOW = '$'):
        super().__init__()
        self.k = k
        # Prefix symbol
        self.EOW = '$'

    def evaluate(self, x, y):
        """
        Evaluation function computing the inner product between phi_x and phi_y
        """

        xwords_occurrence = {}
        count = 0
        for l in range(len(x[0]) - self.k + 1):
            if self.find_word(y[1], x[0][l:l + self.k]):
                if x[0][l:l + self.k] in xwords_occurrence.keys():
                    xwords_occurrence[x[0][l:l + self.k]] += 1
                else:
                    xwords_occurrence[x[0][l:l + self.k]] = 1
                    count += 1

        xphi = np.fromiter(xwords_occurrence.values(), dtype=int)
        yphi = [y[2][key] for key in xwords_occurrence.keys()]

        return xphi.dot(yphi)

    def find_word(self, trie, w):
        """
            Finds whether a word w is present in
            a given retrieval tree "trie"
        """
        tmp = trie
        for l in w:
            if l in tmp:
                tmp = tmp[l]
            else:
                return False
        return self.EOW in tmp

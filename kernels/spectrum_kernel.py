########################################################################
### SpectrumKernel
########################################################################

from kernels.kernel import *

class SpectrumKernel(Kernel):
    """ SpectrumKernel class """

    def __init__(self, k, EOW='$'):
        super().__init__()
        self.k = k
        self.EOW = '$'

    def evaluate(self, x, y):
        xwords = {}
        count = 0
        for l in range(len(x[0]) - self.k + 1):
            if self.find_kmer(y[1], x[0][l:l + self.k]):
                if x[0][l:l + self.k] in xwords.keys():
                    xwords[x[0][l:l + self.k]] += 1
                else:
                    xwords[x[0][l:l + self.k]] = 1
                    count += 1
        xphi = np.fromiter(xwords.values(), dtype=int)
        yphi = [y[2][key] for key in xwords.keys()]

        return xphi.dot(yphi)

    def find_kmer(self, trie, kmer):
        """
            Finds whether a word kmer is present in a given retrieval tree trie
        """

        tmp = trie
        for l in kmer:
            if l in tmp:
                tmp = tmp[l]
            else:
                return False
        return self.EOW in tmp

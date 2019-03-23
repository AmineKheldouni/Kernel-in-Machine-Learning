########################################################################
### SpectrumKernel
########################################################################

from kernels.kernel import *

class SumSpectrumKernel(Kernel):
    """ SumSpectrumKernel class """

    def __init__(self, k_list, EOW = '$'):
        super().__init__()
        self.k_list = k_list
        # Prefix symbol
        self.EOW = '$'

    def evaluate(self, x, y):
        return

    def find_word(self, trie, w):
        return

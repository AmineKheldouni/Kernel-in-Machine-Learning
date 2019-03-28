
import numpy as np
import pandas as pd

from submission import *
from kernels.fast_spectrum_kernel import SpectrumKernel
from kernels.fast_sum_spectrum_kernel import SumSpectrumKernel
#from kernels.linear_kernel import LinearKernel
#from kernels.levenshtein_kernel import LevenshteinKernel
#from kernels.rbf_kernel import RBFKernel
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
import time
from utils import *

from algorithms.svm_mkl import MKL_SVM
from kernels.weighted_sum_kernel import WeightedSumKernel


def SVM_prediction(data_train, data_val, y_train, y_val, kernel, lbd=0.001):

    svm = MKL_SVM(kernel, center=False)
    svm.init_train(data_train, y_train)
    svm.train(data_train, y_train, lbd=lbd, tol = 0.01)

    predictions = np.array(np.sign(svm.predict(data_val)), dtype=int)

    assert (predictions != 0).all()

    return svm, svm.score_train(), 0


# Read training set 0
Xtr0 = pd.read_csv('./data/Xtr0.csv', sep=',', header=0)
Xtr0 = Xtr0['seq'].values

Ytr0 = pd.read_csv('./data/Ytr0.csv', sep=',', header=0)
Ytr0 = Ytr0['Bound'].values
Ytr0 = 2*Ytr0-1

# Read training set 1
Xtr1 = pd.read_csv('./data/Xtr1.csv', sep=',', header=0)
Xtr1 = Xtr1['seq'].values

Ytr1 = pd.read_csv('./data/Ytr1.csv', sep=',', header=0)
Ytr1 = Ytr1['Bound'].values
Ytr1 = 2*Ytr1-1

# Read training set 2
Xtr2 = pd.read_csv('./data/Xtr2.csv', sep=',', header=0)
Xtr2 = Xtr2['seq'].values

Ytr2 = pd.read_csv('./data/Ytr2.csv', sep=',', header=0)
Ytr2 = Ytr2['Bound'].values
Ytr2 = 2*Ytr2-1

X0_train, X0_val, y0_train, y0_val = train_val_split(Xtr0, Ytr0, split = 0)
X1_train, X1_val, y1_train, y1_val = train_val_split(Xtr1, Ytr1, split = 0)
X2_train, X2_val, y2_train, y2_val = train_val_split(Xtr2, Ytr2, split = 0)


Xte0 = pd.read_csv('./data/Xte0.csv', sep=',', header=0)
Xte0 = Xte0['seq'].values

Xte1 = pd.read_csv('./data/Xte1.csv', sep=',', header=0)
Xte1 = Xte1['seq'].values

Xte2 = pd.read_csv('./data/Xte2.csv', sep=',', header=0)
Xte2 = Xte2['seq'].values

##############################################################################
#############################  TRAIN SESSION #################################
##############################################################################

print(">>> Set 0")
# k0 = 5
kernel00 = SpectrumKernel(3, normalize = True)
kernel01 = SpectrumKernel(2, normalize = True)
kernel02 = SpectrumKernel(1, normalize = True)
kernel0 = WeightedSumKernel([kernel00, kernel01, kernel02])

# svm0, train_acc, val_acc = SVM_prediction(X0_train, X0_val, y0_train, y0_val, kernel0, 0.0006)
svm0, train_acc, val_acc = SVM_prediction(Xtr0, Xte0, Ytr0, Ytr0, kernel0, 0.0006)

print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)

###############################################################################
print(">>> Set 1")
kernel00 = MismatchSpectrumKernel(8, 2, normalize = True)
kernel01 = MismatchSpectrumKernel(8, 1, normalize = True)
kernel02 = MismatchSpectrumKernel(8, 0, normalize = True)
kernel03 = MismatchSpectrumKernel(6, 2, normalize = True)
kernel04 = MismatchSpectrumKernel(6, 1, normalize = True)
kernel05 = MismatchSpectrumKernel(6, 0, normalize = True)
kernel06 = MismatchSpectrumKernel(4, 2, normalize = True)
kernel07 = MismatchSpectrumKernel(4, 1, normalize = True)
kernel08 = MismatchSpectrumKernel(4, 0, normalize = True)
kernel1 = WeightedSumKernel([kernel00, kernel01, kernel02, kernel03, kernel04, kernel05, kernel05, kernel06, kernel07, kernel08])

# svm1, train_acc, val_acc = SVM_prediction(X1_train, X1_val, y1_train, y1_val, kernel1, 0.001)
svm1, train_acc, val_acc = SVM_prediction(Xtr1, Xte1, Ytr1, Ytr1, kernel1, 0.001)
print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)

###############################################################################
print(">>> Set 2")

kernel00 = MismatchSpectrumKernel(8, 2, normalize = True)
kernel01 = MismatchSpectrumKernel(8, 1, normalize = True)
kernel02 = MismatchSpectrumKernel(8, 0, normalize = True)
kernel03 = MismatchSpectrumKernel(6, 2, normalize = True)
kernel04 = MismatchSpectrumKernel(6, 1, normalize = True)
kernel05 = MismatchSpectrumKernel(6, 0, normalize = True)
kernel06 = MismatchSpectrumKernel(4, 2, normalize = True)
kernel07 = MismatchSpectrumKernel(4, 1, normalize = True)
kernel08 = MismatchSpectrumKernel(4, 0, normalize = True)
kernel2 = WeightedSumKernel([kernel00, kernel01, kernel02, kernel03, kernel04, kernel05, kernel05, kernel06, kernel07, kernel08])

# svm2, train_acc, val_acc = SVM_prediction(X2_train, X2_val, y2_train, y2_val, kernel2, 0.0008)
svm2, train_acc, val_acc = SVM_prediction(Xtr2, Xte2, Ytr2, Ytr2, kernel2, 0.0008)
print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)


###############################################################################
##############################  TEST SESSION ##################################
###############################################################################




generate_submission_file("Yte_sum_mismatch.csv", svm0, svm1, svm2, Xte0, Xte1, Xte2)


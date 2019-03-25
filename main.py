
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from submission import *
from kernels.fast_spectrum_kernel import SpectrumKernel
from kernels.fast_sum_spectrum_kernel import SumSpectrumKernel
#from kernels.linear_kernel import LinearKernel
#from kernels.levenshtein_kernel import LevenshteinKernel
#from kernels.rbf_kernel import RBFKernel
import time
from utils import *


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

X0_train, X0_val, y0_train, y0_val = train_val_split(Xtr0, Ytr0, split = 300)
X1_train, X1_val, y1_train, y1_val = train_val_split(Xtr1, Ytr1, split = 300)
X2_train, X2_val, y2_train, y2_val = train_val_split(Xtr2, Ytr2, split = 300)

##############################################################################
#############################  TRAIN SESSION #################################
##############################################################################

print(">>> Set 0")
# k0 = 5
k0 = [1,2,3,4,5,6,7,8]
lbd0 = 0.067
# kernel0 = SpectrumKernel(k0,  normalize=True)
kernel0 = SumSpectrumKernel(k0)

svm0, train_acc, val_acc = SVM_prediction(X0_train, X0_val, y0_train, y0_val, kernel0, lbd0)
print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)

###############################################################################
print(">>> Set 1")
k1 = [1,2,3,4,5,6,7,8]
lbd1 = 0.025
kernel1 = SumSpectrumKernel(k1)

svm1, train_acc, val_acc = SVM_prediction(X1_train, X1_val, y1_train, y1_val, kernel1, lbd1)
print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)

###############################################################################
print(">>> Set 2")

lbd2 = 0.098
k2 = [1, 2, 3, 4, 5, 6, 7, 8]
kernel2 = SumSpectrumKernel(k2)

svm2, train_acc, val_acc = SVM_prediction(X2_train, X2_val, y2_train, y2_val, kernel2, lbd2)
print("Training accuracy:", train_acc)
print("Valudation accuracy:", val_acc)


###############################################################################
##############################  TEST SESSION ##################################
###############################################################################


Xte0 = pd.read_csv('./data/Xte0.csv', sep=',', header=0)
Xte0 = Xte0['seq'].values

Xte1 = pd.read_csv('./data/Xte1.csv', sep=',', header=0)
Xte1 = Xte1['seq'].values

Xte2 = pd.read_csv('./data/Xte2.csv', sep=',', header=0)
Xte2 = Xte2['seq'].values

generate_submission_file("Yte_ssk.csv", svm0, svm1, svm2, Xte0, Xte1, Xte2)

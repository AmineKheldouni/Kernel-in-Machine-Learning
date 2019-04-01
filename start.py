
# -*- coding: utf-8 -*-

from imports import *

from kernels.sum_kernel import SumKernel
from kernels.gaussian_kernel import ExponentialLinearKernel
from kernels.spectrum_kernel import SpectrumKernel
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
from kernels.levenshtein_kernel import LevenshteinKernel
from kernels.LA_kernel import LAKernel

from algorithms.svm import SVM

from submission import *
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

X0_train, X0_val, y0_train, y0_val = train_val_split(Xtr0, Ytr0,split=0)
X1_train, X1_val, y1_train, y1_val = train_val_split(Xtr1, Ytr1,split=0)
X2_train, X2_val, y2_train, y2_val = train_val_split(Xtr2, Ytr2,split=0)

##############################################################################
#############################  TRAIN SESSION #################################
##############################################################################

print(">>> Set 0 <<<")
kernel00 = MismatchSpectrumKernel(8, 2, normalize = True)
kernel01 = MismatchSpectrumKernel(8, 1, normalize = True)
kernel02 = MismatchSpectrumKernel(8, 0, normalize = True)
kernel03 = MismatchSpectrumKernel(6, 2, normalize = True)
kernel04 = MismatchSpectrumKernel(6, 1, normalize = True)
kernel05 = MismatchSpectrumKernel(6, 0, normalize = True)
kernel06 = LevenshteinKernel(0.5)
kernel07 = MismatchSpectrumKernel(4, 1, normalize = True)
kernel08 = MismatchSpectrumKernel(4, 0, normalize = True)
kernel0 = SumKernel([kernel00,
                     kernel01,
                     kernel02,
                     kernel03,
                     kernel04,
                     kernel05,
                     kernel06,
                     kernel07,
                     kernel08])
lbd0 = 0.0005

svm0, train_acc, val_acc = SVM_prediction(X0_train, X0_val, y0_train, y0_val, kernel0, lbd0)
print("Training accuracy:", train_acc)
if len(X0_val) > 0:
    print("Validation accuracy:", val_acc)

###############################################################################
print(">>> Set 1 <<<")
kernel00 = MismatchSpectrumKernel(8, 2, normalize = True)
kernel01 = MismatchSpectrumKernel(8, 1, normalize = True)
kernel02 = MismatchSpectrumKernel(8, 0, normalize = True)
kernel03 = MismatchSpectrumKernel(6, 2, normalize = True)
kernel04 = MismatchSpectrumKernel(6, 1, normalize = True)
kernel05 = MismatchSpectrumKernel(6, 0, normalize = True)
kernel06 = LevenshteinKernel(0.5)
kernel07 = MismatchSpectrumKernel(4, 1, normalize = True)
kernel08 = MismatchSpectrumKernel(4, 0, normalize = True)
kernel1 = SumKernel([kernel00,
                     kernel01,
                     kernel02,
                     kernel03,
                     kernel04,
                     kernel05,
                     kernel06,
                     kernel07,
                     kernel08])
lbd1 = 0.0007
svm1, train_acc, val_acc = SVM_prediction(X1_train, X1_val, y1_train, y1_val, kernel1, lbd1)
print("Training accuracy:", train_acc)
if len(X0_val) > 0:
    print("Validation accuracy:", val_acc)

# ###############################################################################
print(">>> Set 2 <<<")

kernel00 = MismatchSpectrumKernel(8, 2, normalize = True)
kernel01 = MismatchSpectrumKernel(8, 1, normalize = True)
kernel02 = MismatchSpectrumKernel(8, 0, normalize = True)
kernel03 = MismatchSpectrumKernel(6, 2, normalize = True)
kernel04 = MismatchSpectrumKernel(6, 1, normalize = True)
kernel05 = MismatchSpectrumKernel(6, 0, normalize = True)
kernel06 = LevenshteinKernel(0.5)
kernel07 = MismatchSpectrumKernel(4, 1, normalize = True)
kernel08 = MismatchSpectrumKernel(4, 0, normalize = True)
kernel2 = SumKernel([kernel00,
                     kernel01,
                     kernel02,
                     kernel03,
                     kernel04,
                     kernel05,
                     kernel06,
                     kernel07,
                     kernel08])
lbd2 = 0.0008
svm2, train_acc, val_acc = SVM_prediction(X2_train, X2_val, y2_train, y2_val, kernel2, lbd2)
print("Training accuracy:", train_acc)
if len(X0_val) > 0:
    print("Validation accuracy:", val_acc)


###############################################################################
##############################  TEST SESSION ##################################
###############################################################################


Xte0 = pd.read_csv('./data/Xte0.csv', sep=',', header=0)
Xte0 = Xte0['seq'].values
print(Xte0.shape)

Xte1 = pd.read_csv('./data/Xte1.csv', sep=',', header=0)
Xte1 = Xte1['seq'].values

Xte2 = pd.read_csv('./data/Xte2.csv', sep=',', header=0)
Xte2 = Xte2['seq'].values

generate_submission_file("Yte.csv", svm0, svm1, svm2, Xte0, Xte1, Xte2)

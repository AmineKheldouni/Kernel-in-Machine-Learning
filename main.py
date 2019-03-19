# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from submission import *
from algorithms.svm import SVM
from kernels.spectrum_kernel import SpectrumKernel
from kernels.linear_kernel import LinearKernel
import time
from utils import *


# Read training set 0
Xtr0 = pd.read_csv('./data/Xtr0.csv', sep=',', header=0)
Xtr0 = Xtr0['seq'].values

Ytr0 = pd.read_csv('./data/Ytr0.csv', sep=',', header=0)
Ytr0 = Ytr0['Bound'].values
# Map the 0/1 labels to -1/1
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

###############################################################################
##############################  TRAIN SESSION #################################
###############################################################################

print(">>> Set 0")
k0 = 5
Xtr0_merged = {}
tries = compute_trie(Xtr0, k0)
occs = compute_occurences(Xtr0, k0)
for i in range(len(Xtr0)):
    Xtr0_merged[i] = (Xtr0[i],tries[i],occs[i])
Xtr0 = Xtr0_merged.copy()

lbd0 = 0.03

svm0 = SVM(SpectrumKernel(k0), center=True)
svm0.train(Xtr0, Ytr0, lbd0)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)

###############################################################################
print(">>> Set 1")
k1 = 7
lbd1 = 0.02
Xtr1_merged = {}
tries = compute_trie(Xtr1, k1)
occs = compute_occurences(Xtr1, k1)
for i in range(len(Xtr0)):
    Xtr1_merged[i] = (Xtr1[i],tries[i],occs[i])
Xtr1 = Xtr1_merged.copy()

svm1 = SVM(SpectrumKernel(k1), center=True)
svm1.train(Xtr1, Ytr1, lbd1)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)

###############################################################################
print(">>> Set 2")

lbd2 = 0.245
k2 = 4
Xtr2_merged = {}
tries = compute_trie(Xtr2, k2)
occs = compute_occurences(Xtr2, k2)
for i in range(len(Xtr0)):
    Xtr2_merged[i] = (Xtr2[i],tries[i],occs[i])
Xtr2 = Xtr2_merged.copy()

svm2 = SVM(SpectrumKernel(k2), center=True)
svm2.train(Xtr2, Ytr2, lbd2)
# Training accuracy
f = svm2.get_training_results()
tmp = Ytr2 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)

###############################################################################
##############################  TEST SESSION ##################################
###############################################################################


Xte0 = pd.read_csv('./data/Xte0.csv', sep=',', header=0)
Xte0 = Xte0['seq'].values

Xte1 = pd.read_csv('./data/Xte1.csv', sep=',', header=0)
Xte1 = Xte1['seq'].values

Xte2 = pd.read_csv('./data/Xte2.csv', sep=',', header=0)
Xte2 = Xte2['seq'].values

generate_submission_file("Yte_spectrum_v0.csv", svm0, svm1, svm2, Xte0, Xte1, Xte2)

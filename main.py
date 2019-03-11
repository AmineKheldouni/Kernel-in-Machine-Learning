# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from submission import *
from algorithms.svm import SVM
from kernels.spectrum_kernel import SpectrumKernel
from kernels.linear_kernel import LinearKernel
import time

def compute_trie(Xtr, k, EOW = '$'):
    n = len(Xtr)
    roots = []
    for i in range(n):
        roots.append({})
        for l in range(len(Xtr[i])-k+1):
            tmp = roots[i]
            for level in range(k):
                tmp = tmp.setdefault(Xtr[i][l+level],{})
            tmp[EOW] = EOW
    return roots

def compute_occurences(Xtr,k):
    n = len(Xtr)
    occs = []
    for i in range(n):
        occs.append({})
        for l in range(len(Xtr[i])-k+1):
            if Xtr[i][l:l+k] in occs[i].keys():
                occs[i][Xtr[i][l:l+k]] += 1
            else:
                occs[i][Xtr[i][l:l+k]] = 1
    return occs

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

n = Xtr0.shape[0]
Xtr0_merged = {}
tries = compute_trie(Xtr0, 5)
occs = compute_occurences(Xtr0, 5)
for i in range(len(Xtr0)):
    Xtr0_merged[i] = (Xtr0[i],tries[i],occs[i])
Xtr0 = Xtr0_merged.copy()

lambd = 0.028

svm0 = SVM(SpectrumKernel(5), center=False)
svm0.train(Xtr0, Ytr0, n, lambd)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)

###############################################################################
print(">>> Set 1")

lambd = 0.025
n = Xtr1.shape[0]
Xtr1_merged = {}
tries = compute_trie(Xtr1, 7)
occs = compute_occurences(Xtr1, 7)
for i in range(len(Xtr0)):
    Xtr1_merged[i] = (Xtr1[i],tries[i],occs[i])
Xtr1 = Xtr1_merged.copy()

svm1 = SVM(SpectrumKernel(7), center=True)
svm1.train(Xtr1, Ytr1, n, lambd)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)

###############################################################################
print(">>> Set 2")

lambd = 0.245

n = Xtr2.shape[0]
Xtr2_merged = {}
tries = compute_trie(Xtr2, 4)
occs = compute_occurences(Xtr2, 4)
for i in range(len(Xtr0)):
    Xtr2_merged[i] = (Xtr2[i],tries[i],occs[i])
Xtr2 = Xtr2_merged.copy()

svm2 = SVM(SpectrumKernel(4), center=True)
svm2.train(Xtr2, Ytr2, n, lambd)
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

generate_submission_file("Yte_spectrum.csv", svm0, svm1, svm2, \
    Xte0, len(Xte0), Xte1, len(Xte1), Xte2, len(Xte2))

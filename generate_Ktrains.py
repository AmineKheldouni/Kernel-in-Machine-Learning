import os
import sys

import numpy as np
import pandas as pd
from submission import *
from kernels.fast_spectrum_kernel import SpectrumKernel
from kernels.fast_sum_spectrum_kernel import SumSpectrumKernel
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
import glob

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

X0_train, X0_val, y0_train, y0_val = train_val_split_fixed(Xtr0, Ytr0,rate=0.8)
X1_train, X1_val, y1_train, y1_val = train_val_split_fixed(Xtr1, Ytr1,rate=0.8)
X2_train, X2_val, y2_train, y2_val = train_val_split_fixed(Xtr2, Ytr2,rate=0.8)


if not os.path.exists('./storage'):
    os.makedirs('./storage')

X_train = [X0_train, X1_train, X2_train]
for d in range(1,3):
    if not os.path.exists('./storage/{}'.format(d)):
        os.makedirs('./storage/{}'.format(d))
    for k in [1,2,3,4,5,6,7,8]:
        for m in [0,1,2,3]:
            print("#################################")
            if "{},{}".format(k,m) in '_'.join(glob.glob('./storage/{}/*.npy'.format(d))):
                print("MSK train of params k={} and m={} already saved".format(k,m))
                continue
            else:
                print("Saving the MSK train of params k={} and m={}".format(k,m))
                kernel = MismatchSpectrumKernel(k,m,normalize=True)
                Ktrain = kernel.compute_train(X_train[d])
                np.save('./storage/{}/{}'.format(str(d), '{},{}'.format(k,m)), Ktrain)
                print("Ktrain saved !")

for d in range(1,3):
    if not os.path.exists('./storage/{}'.format(d)):
        os.makedirs('./storage/{}'.format(d))
    for k in [9,10]:
        for m in [0,1]:
            print("#################################")
            if "{},{}".format(k,m) in '_'.join(glob.glob('./storage/{}/*.npy'.format(d))):
                print("MSK train of params k={} and m={} already saved".format(k,m))
                continue
            else:
                print("Saving the MSK train of params k={} and m={}".format(k,m))
                kernel = MismatchSpectrumKernel(k,m,normalize=True)
                Ktrain = kernel.compute_train(X_train[d])
                np.save('./storage/{}/{}'.format(str(d), '{},{}'.format(k,m)), Ktrain)
                print("Ktrain saved !")

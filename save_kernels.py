from imports import *

from kernels.sum_kernel import SumKernel
from kernels.spectrum_kernel import SpectrumKernel
from kernels.mismatch_spectrum_kernel import MismatchSpectrumKernel
from kernels.gaussian_kernel import ExponentialLinearKernel

from submission import *
from utils import *

kernel_type = sys.argv[1]
i = sys.argv[2]
split = int(sys.argv[3])
params = sys.argv[4]
split_params = params.split(',')

# Read training set i
Xtr = pd.read_csv('./data/Xtr{}.csv'.format(i), sep=',', header=0)
Xtr = Xtr['seq'].values

Ytr = pd.read_csv('./data/Ytr{}.csv'.format(i), sep=',', header=0)
Ytr = Ytr['Bound'].values
Ytr = 2*Ytr-1

X_train, X_val, y_train, y_val = train_val_split(Xtr, Ytr, split = split)


if not os.path.exists('./storage'):
    os.makedirs('./storage')

dic_kernel_names = {'MismatchSpectrumKernel': MismatchSpectrumKernel,
                    'SpectrumKernel': SpectrumKernel,
                    'SumKernel': SumKernel,
                    'ExponentialLinearKernel': ExponentialLinearKernel}

file_name = ['Ktrain']
file_name.append(kernel_type)
file_name.append(str(i))
file_name.append(str(split))

kernel = dic_kernel_names[kernel_type](int(split_params[0]), int(split_params[1]), normalize = True)
file_name.append(params)

print(" Computing training Gram matrix using dataset {}, splitted with {} validation examples.".format(i, str(split)) +
      " The {} is of parameters: {}".format(kernel_type, params))
Ktrain = kernel.compute_train(X_train)
print("Saving training Gram matrix (of shape {}) in storage/{}".format(str(Ktrain.shape), '_'.join(file_name)))
np.save('./storage/'+'_'.join(file_name), Ktrain)
print("Ktrain saved !")

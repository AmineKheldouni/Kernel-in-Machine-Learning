import numpy as np
from cvxopt import matrix, solvers
import math

EPS = math.pow(10,-5)

from cvxopt import matrix, solvers
from svm import SVM
from kernels.weighted_sum_kernel import WeightedSumKernel

class MKL_SVM:
    """
        Implements Support Vector Machine for Multiple Kernels
        http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf
    """

    def __init__(self, kernels=None, center=False):
        self.kernels = kernels
        self.nb_kernels = len(self.kernels)
        self.center = center
        self.etas = 1/self.nb_kernels * np.ones(self.nb_kernels)

    def init_train(self, Xtr, Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Ks = [kernel.compute_train(self.Xtr) for kernel in self.kernels]

    def get_grad_objective(self, alphas):
        pass

    def get_descent_direction(self, grad_target):
        pass

    def train(self, Xtr, Ytr, lbd=1, tol = 0.0001):

        criteria_not_met = True

        while criteria_not_met:

            svm = SVM(kernel = WeightedSumKernel(self.kernels,self.etas))
            svm.init_train(Xtr, Ytr).train(Xtr, Ytr, lbd=lbd)
            target = svm.get_objective()
            grad_target = self.get_grad_objective(svm.alpha)

            #PART 1 : computing optimal descent direction D

            D = self.get_descent_direction(grad_target)
            mu = np.argmax(D)
            new_target  = 0
            new_etas = self.etas
            new_D = D
            step_max = None

            while new_target < target:

                self.etas = new_etas
                D = new_D
                tmp = - self.etas / D
                tmp[np.where(D<0)] = math.inf
                nu = np.argmin(tmp)

                step_max = tmp[nu]
                new_etas += step_max*D
                svm = SVM(kernel=WeightedSumKernel(self.kernels, new_etas))
                svm.init_train(Xtr, Ytr).train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective()

                new_D[mu] = D[mu] - D[nu]
                new_D[nu] = 0

            # PART 2 : computing optimal stepsize

            trials_steps = step_max * np.random.rand(10)
            best_step = None
            best_step_target = - math.inf
            for trial_step in trials_steps:
                svm = SVM(kernel=WeightedSumKernel(self.kernels, new_etas + trial_step * D))
                svm.init_train(Xtr, Ytr).train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective()
                if new_target > best_step_target:
                    best_step = trial_step
                    best_step_target = new_target

            delta_etas = best_step * D
            self.etas += delta_etas
            criteria_not_met =  delta_etas > tol

        # OPTIMIZING

        self.SVM = SVM(kernel=WeightedSumKernel(self.kernels, self.etas))
        self.SVM.init_train(Xtr, Ytr).train(Xtr, Ytr, lbd=lbd)

        print("MKL - SVM solved !")

    def get_training_results(self):
        return self.SVM.get_training_results()

    def predict(self, Xte):
        self.SVM.predict(Xte)

    def score_train(self):
        return self.SVM.score_train()

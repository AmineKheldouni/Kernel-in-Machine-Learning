import numpy as np
from cvxopt import matrix, solvers
import math

EPS = math.pow(10,-5)

from cvxopt import matrix, solvers
from algorithms.svm import SVM
from algorithms.fast_svm import FastSVM
from kernels.weighted_sum_kernel import WeightedSumKernel
import copy

class MKL_SVM:
    """
        Implements Support Vector Machine for Multiple Kernels
        http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf
    """

    def __init__(self, WS_kernel, center=False):
        self.WS_kernel = WS_kernel #WeightedSumKernel(kernels, self.etas)
        self.etas = self.WS_kernel.etas
        self.nb_kernels = self.WS_kernel.nb_kernels
        self.center = center

    def get_grad_objective(self, alphas, Ytr):
        grad = np.zeros(self.nb_kernels)
        tmp = (alphas.flatten() * Ytr)[:,None]
        for i in range(self.nb_kernels):
            grad[i] = tmp.T.dot(self.WS_kernel.K_trains[i]).dot(tmp)[0,0]
        return -0.5 * grad

    def get_descent_direction(self, grad_target, mu):
        descent_directions = np.zeros(self.nb_kernels)
        sum_grad_diff_for_pos_etas = 0
        for i in range(self.nb_kernels):
            grad_diff =  grad_target[i] - grad_target[mu]
            if (self.WS_kernel.etas[i]==0) and (grad_diff>0):
                descent_directions[i] = 0
            elif (self.WS_kernel.etas[i]>0) and (mu !=i ):
                sum_grad_diff_for_pos_etas += grad_diff
                descent_directions[i] = -grad_diff
        descent_directions[mu] = sum_grad_diff_for_pos_etas
        return descent_directions

    def train(self, Xtr, Ytr, lbd=1, tol = 0.01, viz=True):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.WS_kernel.precompute_train(Xtr)

        if viz: print("init etas : ", self.etas)

        mu = 0

        norm_delta_etas = tol + 1
        count_iters = 0

        while norm_delta_etas > tol:

            print("###########################")
            count_iters += 1
            print("Iteration "+str(count_iters)+"\n")

            svm = FastSVM(kernel = self.WS_kernel)
            svm.train(Xtr, Ytr, lbd=lbd)
            target = svm.get_objective(Ytr)
            print("Target: ", target)
            grad_target = self.get_grad_objective(svm.alpha, Ytr)

            #PART 1 : computing optimal descent direction D

            # if viz: print("\n Compute optimal direction D")

            D = self.get_descent_direction(grad_target,mu)
            mu = np.argmax(self.etas)
            step_max = None

            # if viz: print("target to beat : ", target)

            new_etas = self.etas
            new_D = D
            new_target = target - 1

            while new_target < target:

                self.etas = new_etas
                D = new_D
                tmp = - self.etas / D
                tmp[np.where(D>=0)] = math.inf
                nu = np.argmin(tmp)
                print("D",D)
                print("-eta/D",tmp)
                print("nu",nu)

                step_max = tmp[nu]
                print("step_max", step_max)
                if step_max == math.inf:
                    break
                new_etas = self.etas + step_max*D
                self.WS_kernel.etas = new_etas
                svm = FastSVM(kernel = self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective(Ytr)

                new_D[mu] = D[mu] - D[nu]
                new_D[nu] = 0

                if viz: print("updated D, current target : ", new_target)

            if viz: print("target beaten")

            # PART 2 : computing optimal stepsize

            # if viz: print("\n Compute optimal stepsize D")

            trials_steps = np.linspace(0, step_max, 3)
            best_step = None
            best_step_target = -math.inf
            for trial_step in trials_steps:
                # if viz: print("New best stepsize found")
                self.WS_kernel.etas  = self.etas + trial_step * D 
                svm = FastSVM(kernel=self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective(Ytr)
                if new_target >= best_step_target:
                    best_step = trial_step
                    best_step_target = new_target

            delta_etas = best_step * D
            self.etas += delta_etas
            norm_delta_etas = np.linalg.norm(delta_etas)
            if viz: print("Criteria (eta_diff) :  ", norm_delta_etas)

        if viz: print("final etas : ", self.WS_kernel.etas)

        # OPTIMIZING

        self.WS_kernel.etas = self.etas
        self.svm = FastSVM(kernel=self.WS_kernel)
        self.svm.train(Xtr, Ytr, lbd=lbd)

        print("MKL - SVM solved !")

    def get_training_results(self):
        return self.svm.get_training_results()

    def predict(self, Xte):
        return self.svm.predict(Xte)

    def score_train(self):
        return self.svm.score_train()

import numpy as np
import math

from algorithms.utils_mkl import *

EPS = math.pow(10,-5)

#from algorithms.svm import SVM
from algorithms.fast_svm import FastSVM as SVM


def projection_simplex(v):
    #I implemented the linear time projection algorithm onto the simplex designed in the article below :
    #https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf (pivot method)
    U = list(range(v.shape[0]))
    s = 0
    rho = 0
    while len(U)!=0:
        idx_pivot = np.random.choice(U)
        G = []
        L = []
        delta_s = v[idx_pivot]
        for idx in U:
            if idx!=idx_pivot:
                if v[idx]>v[idx_pivot]:
                    G.append(idx)
                    delta_s += v[idx]
                else:
                    L.append(idx)
        delta_rho = len(G) + 1 #1 for v[idx_pivot]
        if (s+delta_s) < 1 + (rho + delta_rho)*v[idx_pivot]:
            s += delta_s
            rho += delta_rho
            U = L
        else:
            U = G
    theta = (s-1)/rho
    res = v-theta
    res[res<0] = 0
    return res


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

    def get_grad_objective(self, alphas):
        grad = np.zeros(self.nb_kernels)
        for i in range(self.nb_kernels):
            grad[i] = alphas.T.dot(self.WS_kernel.K_trains[i]).dot(alphas)[0,0]
        return - 0.5 * grad

    def train(self, Xtr, Ytr, lbd=1, tol = 0.01, viz=True):

        self.Xtr = Xtr
        self.Ytr = Ytr
        self.WS_kernel.precompute_train(Xtr)

        if viz: print("init etas : ", self.etas)

        count_iters = 0
        criteria_not_met = True

        while criteria_not_met:
            print("###########################")
            count_iters += 1
            print("Iteration " + str(count_iters) + "\n")

            svm = SVM(kernel = self.WS_kernel)
            svm.train(Xtr, Ytr, lbd=lbd)
            target = svm.get_objective(Ytr)
            print("Target: ", target)

            #PART 1 : computing optimal descent direction D

            if viz: print("\n Compute optimal direction D")

            grad_target = self.get_grad_objective(svm.alpha)
            mu = self.etas.argmax()
            D = compute_descent_direction(self.etas, grad_target, mu)

            step_max = math.inf
            for m in range(self.nb_kernels):
                if D[m] < 0:
                    d_D_quotient = -1 * self.etas[m] / D[m]
                    if d_D_quotient < step_max:
                        step_max = d_D_quotient

            self.WS_kernel.etas = self.etas + step_max * D
            self.WS_kernel.etas = projection_simplex(self.WS_kernel.etas)
            svm = SVM(kernel=self.WS_kernel)
            svm.train(Xtr, Ytr, lbd=lbd)
            target_cross = svm.get_objective(Ytr)

            # PART 2 : computing optimal stepsize

            if (max(D) < EPS):
                break

            if viz: print("\n Compute optimal stepsize D")

            best_step = step_max
            m = D.T.dot(grad_target)
            divide_again = True
            while divide_again:
                self.WS_kernel.etas = self.etas + best_step * D
                self.WS_kernel.etas = projection_simplex(self.WS_kernel.etas)
                svm = SVM(kernel=self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective(Ytr)
                if new_target <= target_cross + step_max * 0.5 * m:
                    divide_again = False
                else:
                    # Update gamma
                    best_step = best_step * 0.5

            delta_etas = best_step * D
            self.etas += delta_etas
            self.etas = projection_simplex(self.etas)

            print("new etas : ", self.etas)

            criteria_not_met = (max(D) > EPS) and not(stopping_criterion(False,grad_target, self.etas, tol))

        if viz: print("final etas : ", self.WS_kernel.etas)

        # OPTIMIZING

        self.WS_kernel.etas = self.etas.copy()
        self.svm = SVM(kernel=self.WS_kernel)
        self.svm.train(Xtr, Ytr, lbd=lbd)

        print("MKL - SVM solved !")

    def get_training_results(self):
        return self.svm.get_training_results()

    def predict(self, Xte):
        return self.svm.predict(Xte)

    def score_train(self):
        return self.svm.score_train()



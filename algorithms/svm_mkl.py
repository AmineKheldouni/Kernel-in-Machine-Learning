import numpy as np
import math

from algorithms.utils_mkl import *

EPS = math.pow(10,-5)

#from algorithms.svm import SVM
from algorithms.fast_svm import FastSVM as SVM


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
            D = fix_precision_of_vector(D, 0)

            target_cross = -math.inf
            etas_cross = self.etas.copy()
            D_cross = D.copy()

            step_max = math.inf

            while target_cross < target:

                self.etas = etas_cross.copy()
                D = D_cross.copy()

                self.WS_kernel.etas = self.etas.copy()
                svm = SVM(kernel=self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                #target_cross = svm.get_objective(Ytr)

                step_max = math.inf
                nu = None
                for m in range(self.nb_kernels):
                    if D[m] < 0:
                        d_D_quotient = -1 * self.etas[m] / D[m]
                        if d_D_quotient < step_max:
                            step_max = d_D_quotient
                            nu = m

                if (max(D) < EPS):
                    print("D all zero")
                    break

                etas_cross = self.etas + step_max * D

                etas_cross[nu] = 0
                print("D", D)
                print("Dmu ", D[mu])
                print("Dnu ", D[nu])
                print("Dmu + Dnu ", D[mu] + D[nu])

                D_cross[mu] = D[mu] + D[nu] #should not be minus
                D_cross[nu] = 0

                etas_cross = fix_precision_of_vector(etas_cross, 1)
                D_cross = fix_precision_of_vector(D_cross, 0)
                print("bis", D_cross[mu])

                self.WS_kernel.etas = etas_cross.copy()
                svm = SVM(kernel=self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                target_cross = svm.get_objective(Ytr)

            # PART 2 : computing optimal stepsize

            if (max(D) < EPS):
                break

            if viz: print("\n Compute optimal stepsize D")

            trials_steps = np.linspace(0, step_max, 3)
            best_step = None
            best_step_target = -math.inf
            for trial_step in trials_steps:
                # if viz: print("New best stepsize found")
                self.WS_kernel.etas = self.etas + trial_step * D
                self.WS_kernel.etas = fix_precision_of_vector(self.WS_kernel.etas, 1)
                svm = SVM(kernel=self.WS_kernel)
                svm.train(Xtr, Ytr, lbd=lbd)
                new_target = svm.get_objective(Ytr)
                if new_target >= best_step_target:
                    best_step = trial_step
                    best_step_target = new_target

            # gamma = helpers.get_armijos_step_size(kernel_matrices, d, y_mat, alpha,
            #                                  box_constraints, gamma_max, J_cross,
            #D, dJ)

            delta_etas = best_step * D
            self.etas += delta_etas
            self.etas = fix_precision_of_vector(self.etas, 1)

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

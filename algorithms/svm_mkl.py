import numpy as np
import math

EPS = math.pow(10,-2)

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

    def get_descent_direction(self, grad_target, mu):
        descent_directions = np.zeros(self.nb_kernels)
        sum_grad_diff_for_pos_etas = 0
        for i in range(self.nb_kernels):
            grad_diff = grad_target[i] - grad_target[mu]
            if (self.WS_kernel.etas[i] == 0) and (grad_diff > 0):
                descent_directions[i] = 0
            elif (self.WS_kernel.etas[i] > 0) and (mu != i):
                sum_grad_diff_for_pos_etas += grad_diff
                descent_directions[i] = -grad_diff
        descent_directions[mu] = sum_grad_diff_for_pos_etas
        return descent_directions

    def train(self, Xtr, Ytr, lbd=1, tol = 0.1, viz=False):

        self.Xtr = Xtr
        self.Ytr = Ytr
        self.WS_kernel.precompute_train(Xtr)

        if viz: print("init etas : ", self.etas)

        count_iters = 0
        criteria_not_met = True

        while criteria_not_met:

            old_etas = self.etas.copy()

            print("###########################")
            count_iters += 1
            print("Iteration " + str(count_iters) + "\n")

            svm = SVM(kernel = self.WS_kernel)
            svm.train(Xtr, Ytr, lbd=lbd, verbose=viz)
            target = svm.get_objective(Ytr)
            print("Target: ", target)

            #PART 1 : computing optimal descent direction D

            if viz: print("\n Compute optimal direction D")

            grad_target = self.get_grad_objective(svm.alpha)
            mu = self.etas.argmax()
            D = self.get_descent_direction(grad_target, mu)

            if False: #we often reach mu=nu which is really unexepected !

                # PART 1 bis : computing optimal descent direction D

                target_cross = -math.inf
                etas_cross = self.etas.copy()
                D_cross = D.copy()

                step_max = math.inf

                while target_cross < target:

                    self.etas = etas_cross.copy()
                    D = D_cross.copy()

                    self.WS_kernel.etas = self.etas.copy()
                    svm = SVM(kernel=self.WS_kernel)
                    svm.train(Xtr, Ytr, lbd=lbd, verbose=viz)
                    #target_cross = svm.get_objective(Ytr)

                    step_max = math.inf
                    nu = None
                    for m in range(self.nb_kernels):
                        if D[m] < 0:
                            d_D_quotient = -1 * self.etas[m] / D[m]
                            if d_D_quotient < step_max:
                                step_max = d_D_quotient
                                nu = m

                    if viz:
                        print("Gradient Dir D", D)
                        print("Dmu ", D[mu])
                        print("Dnu ", D[nu])

                    if (max(D) < EPS):
                        if viz: print("D all zero")
                        break

                    etas_cross = self.etas + step_max * D

                    #etas_cross[nu] = 0
                    if mu!=nu:
                        if viz: print(D_cross)
                        D_cross[mu] = D[mu] + D[nu] #should not be minus
                        D_cross[nu] = 0
                        if viz: print(D_cross)
                    else:
                        if viz: print("mu=nu")
                        break

                    self.WS_kernel.etas = etas_cross.copy()
                    svm = SVM(kernel=self.WS_kernel)
                    svm.train(Xtr, Ytr, lbd=lbd, verbose=viz)
                    target_cross = svm.get_objective(Ytr)

                # PART 2 : computing optimal stepsize

                if (max(D) < EPS):
                    break

                if viz: print("\n Compute optimal stepsize D")

                best_step = step_max
                m = D.T.dot(grad_target)
                divide_again = True
                if viz: print("step max :", best_step)
                counter_armijo = 0
                while divide_again:
                    self.WS_kernel.etas = self.etas + best_step * D
                    svm = SVM(kernel=self.WS_kernel)
                    svm.train(Xtr, Ytr, lbd=lbd, verbose=viz)
                    new_target = svm.get_objective(Ytr)
                    if new_target <= target_cross + step_max * 0.5 * m:
                        divide_again = False
                    else:
                        # Update gamma
                        best_step = best_step * 0.5
                    counter_armijo += 1
                    if counter_armijo>10:
                        best_step = 0
                        break
                if viz: print("chosen step (Armijo) :", best_step)

            else: #fix stepsize
              best_step = 0.000005

            delta_etas = best_step * D
            self.etas += delta_etas

            print("new etas : ", self.etas)

            criteria_not_met = np.linalg.norm(old_etas-self.etas) > EPS

        print("final etas : ", self.WS_kernel.etas)

        # OPTIMIZING

        self.WS_kernel.etas = self.etas
        self.svm = SVM(kernel=self.WS_kernel)
        self.svm.train(Xtr, Ytr, lbd=lbd, verbose=viz)

        print("MKL - SVM solved !")

    def get_training_results(self):
        return self.svm.get_training_results()

    def predict(self, Xte):
        return self.svm.predict(Xte)

    def score_train(self):
        return self.svm.score_train()


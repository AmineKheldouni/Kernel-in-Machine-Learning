import numpy as np

from algorithms.svm import SVM
from algorithms.fast_svm import FastSVM

def SVM_prediction(data_train, data_val, y_train, y_val, kernel, lbd=0.001, svm_function=SVM):

    svm = svm_function(kernel, center=False)
    svm.train(data_train, y_train, lbd)

    predictions = np.array(np.sign(svm.predict(data_val)), dtype=int)

    assert (predictions != 0).all()

    return svm, svm.score_train(), (y_val == predictions).mean()

def train_val_split(data, y, split = 300):
    if split == 0:
        return data, np.array([]), y, np.array([])
    permutation = np.random.permutation(len(data))
    y_train = y[permutation][:-split]
    y_test = y[permutation][-split:]
    data_train = data[permutation][:-split]
    data_test = data[permutation][-split:]
    return data_train, data_test, y_train, y_test

def kfold_cross_validation(data, predictions, kernels, list_lambda, k=10):

    nfold = len(data) // k
    scores = np.zeros((k, len(list_lambda), 3))
    for d in range(3):
        permutation = np.random.permutation(len(data[d]))
        predictions_bucket = list(predictions[d][permutation])
        data_bucket = list(data[d][permutation])
        for j in range(len(list_lambda)):
            for i in range(k):
                print("##############################")
                print("Performing Fold {} (lambda={})".format(i+1,list_lambda[j]))
                data_val = np.array(data_bucket[i*nfold:(i+1)*nfold])
                y_val = np.array(predictions_bucket[i*nfold:(i+1)*nfold])
                data_train = np.array(data_bucket[:i*nfold] + data_bucket[(i+1)*nfold:])
                y_train = np.array(predictions_bucket[:i*nfold] + predictions_bucket[(i+1)*nfold:])
                svm, train_acc, val_acc = SVM_prediction(data_train,
                                                         data_val,
                                                         y_train,
                                                         y_val,
                                                         kernels[d],
                                                         list_lambda[j],
                                                         svm_function=FastSVM)
                print("Training accuracy:", train_acc)
                print("Valudation accuracy:", val_acc)
                print("######")
                scores[i,j,d] = val_acc

    print("####################################")
    return scores, np.mean(scores,axis=0)

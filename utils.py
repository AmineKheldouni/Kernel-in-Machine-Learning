import numpy as np
from algorithms.svm import SVM

def SVM_prediction(data_train, data_val, y_train, y_val, kernel, lbd=0.001):

    svm = SVM(kernel, center=False)
    svm.train(data_train, y_train, lbd)

    predictions = np.array(np.sign(svm.predict(data_val)), dtype=int)

    assert (predictions != 0).all()

    return svm, svm.score_train(), (y_val == predictions).mean()

def train_val_split(data, y, split = 300):
    permutation = np.random.permutation(len(data))
    y_train = y[permutation][:-split]
    y_test = y[permutation][-split:]
    data_train = data[permutation][:-split]
    data_test = data[permutation][-split:]
    return data_train, data_test, y_train, y_test

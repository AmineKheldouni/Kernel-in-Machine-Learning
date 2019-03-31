import numpy as np
from algorithms.svm import SVM

def SVM_prediction(data_train, data_val, y_train, y_val, kernel, lbd=0.001):
    """ Function for validation of the chosen kernel:
        - Training the kernel with (data_train, y_train).
        - Validation of the kernel for the validation.
        - Displaying the accuracies """
    np.random.seed(0)
    svm = SVM(kernel, center=False)
    svm.train(data_train, y_train, lbd)
    predictions = np.array(np.sign(svm.predict(data_val)), dtype=int)
    assert (predictions != 0).all()
    return svm, svm.score_train(), (y_val == predictions).mean()

def train_val_split(data, y, split = 300):
    """ Function for splitting the data into train and validation sets. """
    if split == 0:
        return data, np.array([]), y, np.array([])
    permutation = np.random.permutation(len(data))
    y_train = y[permutation][:-split]
    y_test = y[permutation][-split:]
    data_train = data[permutation][:-split]
    data_test = data[permutation][-split:]
    return data_train, data_test, y_train, y_test

def kfold_cross_validation(data, predictions, kernels, list_lambda, k=10):
    """
        This function is for cross-validation :
        It divides the dataset into k buckets of examples. Then the validation
        is performed k-times for each of the buckets and the score is given as the
        averaged accuracy over the k-buckets. This is mainly done to tune the
        regularization term (lambda).
    """
    nfold = len(data[0]) // k
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
                print("data_val shape {}".format(data_val.shape))
                y_val = np.array(predictions_bucket[i*nfold:(i+1)*nfold])
                data_train = np.array(data_bucket[:i*nfold] + data_bucket[(i+1)*nfold:])
                y_train = np.array(predictions_bucket[:i*nfold] + predictions_bucket[(i+1)*nfold:])
                svm, train_acc, val_acc = SVM_prediction(data_train,
                                                         data_val,
                                                         y_train,
                                                         y_val,
                                                         kernels[d],
                                                         list_lambda[j])
                print("Training accuracy:", train_acc)
                print("Valudation accuracy:", val_acc)
                print("######")
                scores[i,j,d] = val_acc

    print("####################################")
    return scores, np.mean(scores,axis=0)

def train_val_split_fixed(data, y, rate=0.8):
    n =  len(data)
    return data[:int(rate*n)], data[int(rate*n):], y[:int(rate*n)], y[int(rate*n):]

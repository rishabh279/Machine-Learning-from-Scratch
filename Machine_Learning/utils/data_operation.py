import math
import numpy as np


def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p*log2(p)
    return entropy


def calculate_variance(x):
    mean = np.ones(np.shape(x)) * x.mean(0)
    n_samples = np.shape(x)[0]
    variance = (1 / n_samples) * np.diag((x - mean).T.dot(x - mean))
    return variance


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def mean_squared_error(y_true, y_pred):
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


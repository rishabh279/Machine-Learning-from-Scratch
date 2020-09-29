import numpy as np


def one_hot_conversion(y):
    """Convert the given label into one hot representation"""
    n = len(y)
    k = len(set(y))
    encoded_values = np.zeros((n, k))
    encoded_values[np.arange(n), y] = 1
    return encoded_values


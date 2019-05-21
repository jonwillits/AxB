import numpy as np


def calc_cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


# tmp = np.random.binomial(n=1, p=0.5, size=(10, 10))
# print(tmp)
# print(tmp.sum(axis=1))
# print(np.expand_dims(tmp.sum(axis=1), axis=1))
#
#
# x = tmp / np.expand_dims(tmp.sum(axis=1), axis=1)
# y = np.eye(x.shape[0])
# print(x.round(2))
# print(x.sum(axis=1).round(2))
#
#
#
#
# print(calc_cross_entropy(x, y))
# print(np.exp(calc_cross_entropy(x, y)))
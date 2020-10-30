import numpy as np


def covw(center_X, center_Y, w):
    n = w.shape[1]
    sqrt_w = np.sqrt(w)
    sum_w = w.sum()
    V = np.concatenate((center_X, center_Y), axis=0)
    V = sqrt_w * V
    dis = np.dot(V, V.T) / sum_w * (n / (n - 1))

    return dis

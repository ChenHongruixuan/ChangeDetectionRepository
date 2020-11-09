import numpy as np
import cv2 as cv

import gdal
from skfuzzy import cmeans
from scipy.spatial.distance import cdist
import time


def KI(data, step=255):
    min_value = np.min(data)
    max_value = np.max(data)
    unit_value = (max_value - min_value) / step
    T = min_value + unit_value
    best_J = -np.inf
    best_T = min_value
    while T < max_value:
        data_1 = data[data <= T]
        data_2 = data[data > T]
        w1 = data_1.shape[0]
        w2 = data_2.shape[0]
        mean_1 = data_1.mean()
        mean_2 = data_2.mean()
        var_1 = (data_1 - mean_1).var()
        var_2 = (data_2 - mean_2).var()
        J = 1 + 2 * (w1 * (np.log(var_1) - np.log(w1)) + w2 * (np.log(var_2) - np.log(w2)))
        if J > best_J:
            best_J = J
            best_T = T
        T += unit_value

    bwp = np.zeros(data.shape)
    bwp[data <= best_T] = 0
    bwp[data > best_T] = 255
    print('KI is done')
    return bwp, best_T


def otsu(data, num=400, get_bcm=False):
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :param get_bcm: bool, get bcm or not
    :return:
        binary change map
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    if get_bcm:
        bwp = np.zeros(data.shape)
        bwp[data <= best_threshold] = 0
        bwp[data > best_threshold] = 255
        print('otsu is done')
        return bwp, best_threshold
    else:
        print('otsu is done')
        return best_threshold


def fuzzy_c_means(data, m=2, types=2, max_iter=1000, epsilon=1e-6):
    """
    generate binary change map based on FCM
    :param m:
    :param types:
    :param data: cluster data, (1, data_item)
    :param max_iter: intensity number
    :param epsilon: bool, get bcm or not
    :return:
        binary change map
    """
    print('start fuzzy c-means!')
    start_time = time.time()
    data_item = data.shape[1]
    np.random.seed()
    U = np.random.rand(types, data_item)
    U = U / np.sum(U, axis=0, keepdims=True)  # normalization

    repeat_data = np.tile(data, (types, 1))
    expo = -2. / (m - 1)
    for _iter in range(max_iter):
        last_U = np.copy(U)
        # calculate the c cluster centers v with U
        sum_d = np.sum(np.power(U, m) * data, axis=1, keepdims=True)  # (c, data_item)
        sum_w = np.sum(np.power(U, m), axis=1, keepdims=True)
        ctrl = sum_d / sum_w  # (c, 1)
        # calculate the membership matrix U

        distance = np.abs(repeat_data - ctrl)  # (c, data_item)
        distance = distance.astype(dtype=np.float64)

        distance /= np.max(distance, axis=0, keepdims=True)
        distance /= np.min(distance, axis=0, keepdims=True)
        distance = np.power(distance, expo)
        U = distance / np.sum(distance, axis=0, keepdims=True)

        if np.max(np.abs(U - last_U)) < epsilon:
            break

    class_type = np.argmax(U, axis=0)

    class_value_mean = np.array([np.mean(data[:, class_type == 0]), np.mean(data[:, class_type == 1]),
                                 np.mean(data[:, class_type == 2]), np.mean(data[:, class_type == 3])])
    sort_idx = class_value_mean.argsort()

    bwp = np.zeros(data.shape)

    bwp[:, class_type == sort_idx[0]] = 0
    bwp[:, class_type == sort_idx[1]] = 85
    bwp[:, class_type == sort_idx[2]] = 170
    bwp[:, class_type == sort_idx[3]] = 255
    cost_time = time.time() - start_time
    print('fuzzy c-means is done, cost time {:.4}s'.format(cost_time))

    return bwp, class_type


def local_info_fuzzy_c_means(data, m=2, types=2, max_iter=1000, epsilon=1e-6):
    pass

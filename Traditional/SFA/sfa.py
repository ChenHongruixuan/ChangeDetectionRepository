"""
Slow feature analysis
C. Wu, B. Du, and L. Zhang, “Slow feature analysis for change detection in multispectral imagery,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 5, pp. 2858–2874, 2014.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.stats import chi2
from sklearn.cluster import KMeans

from compare_method.otsu import otsu


def get_binary_change_map(data, method='k_means'):
    """
    get binary change map
    :param data:
    :param method: cluster method
    :return: binary change map
    """
    if method == 'k_means':
        cluster_center = KMeans(n_clusters=2, max_iter=1500).fit(data.T).cluster_centers_.T  # shape: (1, 2)
        # cluster_center = k_means_cluster(weight, cluster_num=2)
        print('k-means cluster is done, the cluster center is ', cluster_center)
        dis_1 = np.linalg.norm(data - cluster_center[0, 0], axis=0, keepdims=True)
        dis_2 = np.linalg.norm(data - cluster_center[0, 1], axis=0, keepdims=True)

        bcm = np.copy(data)  # binary change map
        if cluster_center[0, 0] > cluster_center[0, 1]:
            bcm[dis_1 > dis_2] = 0
            bcm[dis_1 <= dis_2] = 255
        else:
            bcm[dis_1 > dis_2] = 255
            bcm[dis_1 <= dis_2] = 0
    elif method == 'otsu':
        bcm, threshold = otsu(data, num=200)
        print('otsu is done, the threshold is ', threshold)

    return bcm


class SFA(object):
    def __init__(self, img_X, img_Y, data_format='CHW'):
        """
        the init function
        :param img_X: former temporal image, its dim is (band_count, width, height)
        :param img_Y: latter temporal image, its dim is (band_count, width, height)
        """
        if data_format == 'HWC':
            self.img_X = np.transpose(img_X, [2, 0, 1])
            self.img_Y = np.transpose(img_Y, [2, 0, 1])
        else:
            self.img_X = img_X
            self.img_Y = img_Y

        channel, height, width = self.img_X.shape
        self.L = np.zeros((channel - 2, channel))  # (C-2, C)
        for i in range(channel - 2):
            self.L[i, i] = 1
            self.L[i, i + 1] = -2
            self.L[i, i + 2] = 1
        self.Omega = np.dot(self.L.T, self.L)  # (C, C)
        self.norm_method = ['LSR', 'NR', 'OR']

    def isfa(self, max_iter=30, epsilon=1e-6, norm_trans=False, regular=False):
        """
         extract change and unchange info of temporal images based on USFA
         if max_iter == 1, ISFA is equal to SFA
        :param max_iter: the maximum count of iteration
        :param epsilon: convergence threshold
        :param norm_trans: whether normalize the transformation matrix
        :return:
            ISFA_variable: ISFA variable, its dim is (band_count, width * height)
            lamb: last lambda
            all_lambda: all lambda in convergence process
            trans_mat: transformation matrix
            T: last IWD, if max_iter == 1, T is chi-square distance
            weight: the unchanged probability of each pixel
        """

        bands_count, img_height, img_width = self.img_X.shape
        P = img_height * img_width
        # row-major order after reshape
        img_X = np.reshape(self.img_X, (-1, img_height * img_width))  # (band, width * height)
        img_Y = np.reshape(self.img_Y, (-1, img_height * img_width))  # (band, width * height)
        lamb = 100 * np.ones((bands_count, 1))
        all_lambda = []
        weight = np.ones((img_width, img_height))  # (1, width * height)
        # weight[302:343, 471] = 1  # init seed
        # weight[209, 231:250] = 1
        # weight[335:362, 570] = 1
        # weight[779, 332:387] = 1

        weight = np.reshape(weight, (-1, img_width * img_height))
        for _iter in range(max_iter):
            sum_w = np.sum(weight)
            mean_X = np.sum(weight * img_X, axis=1, keepdims=True) / np.sum(weight)  # (band, 1)
            mean_Y = np.sum(weight * img_Y, axis=1, keepdims=True) / np.sum(weight)  # (band, 1)
            center_X = (img_X - mean_X)
            center_Y = (img_Y - mean_Y)

            # cov_XY = covw(center_X, center_Y, weight)  # (2 * band, 2 * band)
            # cov_X = cov_XY[0:bands_count, 0:bands_count]
            # cov_Y = cov_XY[bands_count:2 * bands_count, bands_count:2 * bands_count]
            var_X = np.sum(weight * np.power(center_X, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            var_Y = np.sum(weight * np.power(center_Y, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            std_X = np.reshape(np.sqrt(var_X), (bands_count, 1))
            std_Y = np.reshape(np.sqrt(var_Y), (bands_count, 1))

            # normalize image
            norm_X = center_X / std_X
            norm_Y = center_Y / std_Y
            diff_img = (norm_X - norm_Y)
            mat_A = np.dot(weight * diff_img, diff_img.T) / ((P - 1) * sum_w / P)
            mat_B = (np.dot(weight * norm_X, norm_X.T) +
                     np.dot(weight * norm_Y, norm_Y.T)) / (2 * (P - 1) * sum_w / P)
            if regular:
                penalty = np.trace(mat_B) / np.trace(self.Omega)
                mat_B += penalty * self.Omega
            # solve generalized eigenvalue problem and get eigenvalues and eigenvector
            eigenvalue, eigenvector = eig(mat_A, mat_B)
            eigenvalue = eigenvalue.real  # discard imaginary part
            idx = eigenvalue.argsort()
            eigenvalue = eigenvalue[idx]

            # make sure the max absolute value of vector is 1,
            # and the final result will be more closer to the matlab result
            aux = np.reshape(np.abs(eigenvector).max(axis=0), (1, bands_count))
            eigenvector = eigenvector / aux

            # print sqrt(lambda)
            if (_iter + 1) == 1:
                print('sqrt lambda:')
            print(np.sqrt(eigenvalue))

            eigenvalue = np.reshape(eigenvalue, (bands_count, 1))  # (band, 1)
            threshold = np.max(np.abs(np.sqrt(lamb) - np.sqrt(eigenvalue)))
            # if sqrt(lambda) converge
            if threshold < epsilon:
                break
            lamb = eigenvalue
            all_lambda = lamb if (_iter + 1) == 1 else np.concatenate((all_lambda, lamb), axis=1)
            # the order of the slowest features is determined by the order of the eigenvalues
            trans_mat = eigenvector[:, idx]
            # satisfy the constraints(3)
            if norm_trans:
                output_signal_std = 1 / np.sqrt(np.diag(np.dot(trans_mat.T, np.dot(mat_B, trans_mat))))
                trans_mat = output_signal_std * trans_mat
            ISFA_variable = np.dot(trans_mat.T, norm_X) - np.dot(trans_mat.T, norm_Y)

            if (_iter + 1) == 1:
                T = np.sum(np.square(ISFA_variable) / np.sqrt(lamb), axis=0, keepdims=True)  # chi square
            else:
                T = np.sum(np.square(ISFA_variable) / np.sqrt(lamb), axis=0, keepdims=True)  # IWD
            weight = 1 - chi2.cdf(T, bands_count)

        if (_iter + 1) == max_iter:
            print('the lambda may not be converged')
        else:
            print('the lambda is converged, the iteration is %d' % (_iter + 1))

        return ISFA_variable, lamb, all_lambda, trans_mat, T, weight

    def draw_lambda(self, lambda_list, sqrt=True):
        """
        draw all lambda tendency
        :param lambda_list: list contains all lambda
        :param sqrt: if draw sqrt(lambda)
        :return:
        """
        plt.title('sqrt(lambda) over the iteration')
        plt.ylabel('sqrt(lambda)')
        plt.xlabel('iteration')
        i = 0
        for lamb in lambda_list:
            i += 1
            if sqrt:
                plt.plot(np.sqrt(lamb), '-*', label='band' + str(i))
            else:
                plt.plot(lamb, '-*', label='band' + str(i))
        plt.legend(loc='lower right')
        plt.show()

    def stretch_band(self, bands):
        """
        stretch bands' value
        :param bands: band data
        :return:
            stretched bands
        """
        stretched_band = []
        band_count = bands.shape[0]
        for band in range(band_count):
            min_value = bands[band].min()
            max_value = bands[band].max()
            diff = max_value - min_value
            multi = 255.0 / diff
            new_band = np.array(multi * (bands[band] - min_value), dtype=np.uint8)
            stretched_band.append(new_band)
        return np.array(stretched_band)

    def draw_distribution(self, variable, bins=100, alpha=0.75):
        """
        draw variable's histogram and show its distribution
        :param variable: drawn variable
        :param bins: step
        :param alpha: transparency
        :return:
        """
        plt.title('variable distribution')
        plt.ylabel('count')
        plt.xlabel('value')
        f_var = variable.flatten()
        n, bins, patches = plt.hist(f_var, bins=bins, density=0, facecolor='blue', alpha=alpha)
        plt.show()

    def radio_norm(self, target_img, ref_img, weight, method='LSR'):
        if not (method.upper() in self.norm_method):
            print('No method!!!!!')
            return
        bands_count, img_height, img_width = target_img.shape
        P = img_height * img_width

        target_img = np.reshape(target_img, (-1, P))
        ref_img = np.reshape(ref_img, (-1, P))

        sum_w = weight.sum()
        mean_X = np.sum(weight * target_img, axis=1, keepdims=True) / sum_w
        mean_Y = np.sum(weight * ref_img, axis=1, keepdims=True) / sum_w
        var_X = np.sum(weight * np.square((target_img - mean_X)), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
        var_Y = np.sum(weight * np.square((ref_img - mean_Y)), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
        cov_XY = np.sum(weight * (target_img - mean_X) * (ref_img - mean_Y), axis=1, keepdims=True) / (
                (P - 1) * sum_w / P)

        # three method
        # LSR, NR, OR
        a1 = cov_XY / var_X
        a2 = mean_Y - a1 * mean_X

        b1 = np.sqrt(var_Y / var_X)
        b2 = mean_Y - b1 * mean_X

        c1 = ((var_Y - var_X) + np.sqrt(np.square(var_Y - var_X) + 4 * np.square(cov_XY))) / (2 * cov_XY)
        c2 = mean_Y - c1 * mean_X

        if method == 'LSR':
            normTarImg = a1 * target_img + a2
        elif method == 'NR':
            normTarImg = b1 * target_img + b2
        elif method == 'OR':
            normTarImg = c1 * target_img + c2
        return normTarImg, method

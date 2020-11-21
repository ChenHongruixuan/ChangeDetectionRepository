import imageio

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score


def accuracy_assessment(gt_changed, gt_unchanged, changed_map):
    """
   assess accuracy of changed map based on ground truth
   :param gt_changed: changed ground truth
   :param gt_unchanged: unchanged ground truth
   :param changed_map: changed map
   :return: confusion matrix and overall accuracy
   """
    cm = []

    gt = []

    height, width = changed_map.shape
    changed_map = np.reshape(changed_map, (-1,))
    gt_changed = np.reshape(gt_changed, (-1,))
    gt_unchanged = np.reshape(gt_unchanged, (-1,))

    cm = np.ones((height * width,))
    cm[changed_map == 255] = 2

    gt = np.zeros((height * width,))
    gt[gt_changed == 255] = 2
    gt[gt_unchanged == 255] = 1

    conf_mat = confusion_matrix(y_true=gt, y_pred=cm,
                                labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm,
                                 labels=[1, 2])

    oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return conf_mat, oa, kappa_co


if __name__ == '__main__':
    ground_truth_changed = imageio.imread('../../Dataset/Landsat/Taizhou/change.bmp')
    ground_truth_unchanged = imageio.imread('../../Dataset/Landsat/Taizhou/unchanged.bmp')

    cm_path = '../Traditional/MAD/result/IRMAD_Taizhou.png'
    changed_map = imageio.imread(cm_path)

    conf_mat, oa, kappa_co = accuracy_assessment(ground_truth_changed, ground_truth_unchanged, changed_map)
    print(f'overall accuracy is {oa}, kappa coefficient is {kappa_co}')

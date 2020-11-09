import gdal
import numpy as np
from Methodology.Traditional.PCAKmeans.algorithm import pca_k_means
from Methodology.Traditional.PCAKmeans.util import diff_image
import imageio


def main():
    before_img = imageio.imread('../../../Dataset/PCAKmeans/burn_1986.png')[:, :, 0:3]
    after_img = imageio.imread('../../../Dataset/PCAKmeans/burn_1992.png')[:, :, 0:3]
    eig_dim = 10
    block_sz = 4

    diff_img = diff_image(before_img, after_img, is_abs=True, is_multi_channel=True)
    change_img = pca_k_means(diff_img, block_size=block_sz,
                             eig_space_dim=eig_dim)
    imageio.imwrite('PCAKmeans_burn.png', change_img)


if __name__ == '__main__':
    main()

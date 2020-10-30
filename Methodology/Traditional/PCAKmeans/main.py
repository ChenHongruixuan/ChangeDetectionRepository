import cv2
import gdal
import numpy as np
from PCA_K_menas.algorithm import pca_k_means
from util import gene_gauss_noisy_image, show_img_list, diff_image

def main():
    result_folder = 'result/'
   
    before_img = cv2.imread(img_folder + 'burn_1986.png')
    after_img = cv2.imread(img_folder + 'burn_1992.png')
    max_eig_dim = 10
    block_sz = 4
    bands_count = before_img.shape[2]
    for band_id in range(0, bands_count):
        for eig_dim in range(1, max_eig_dim + 1):
            img_name = 'band' + str(band_id + 1) + '_H' + str(4) + '_S' + str(eig_dim)

            diff_img = diff_image(before_img[:, :, band_id], after_img[:, :, band_id], is_abs=True)
            change_img = pca_k_means(diff_img, block_size=4,
                                         eig_space_dim=eig_dim, image_name=img_name)
            cv2.imwrite(result_folder + img_name + '.bmp', change_img)


if __name__ == '__main__':
   main()
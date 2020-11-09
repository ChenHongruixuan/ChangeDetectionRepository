import numpy as np
import imageio
import gdal
import time
from Methodology.util.cluster_util import otsu
from Methodology.util.data_prepro import stad_img


def CVA(img_X, img_Y, stad=False):
    # CVA has not affinity transformation consistency, so it is necessary to normalize multi-temporal images to
    # eliminate the radiometric inconsistency between them
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm


def main():
    data_set_X = gdal.Open('../../../Dataset/Landsat/Taizhou/2000TM')  # data set X
    data_set_Y = gdal.Open('../../../Dataset/Landsat/Taizhou/2003TM')  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y)

    bcm = np.ones((img_height, img_width))
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    imageio.imwrite('CVA_Taizhou.png', bcm)
    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()

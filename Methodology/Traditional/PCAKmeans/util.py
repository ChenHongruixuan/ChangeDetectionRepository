import numpy as np
import cv2 as cv


def diff_image(image_before, image_after, is_abs=True):
    if is_abs:
        img_diff = np.abs(np.array(image_before, dtype=np.float32) - np.array(image_after, dtype=np.float32))
    else:
        img_diff = np.array(image_before, dtype=np.float32) - np.array(image_after, dtype=np.float32)
    return np.array(img_diff, dtype=np.uint8)


def zero_pad(img, pad):
    """
     Pad with zeros all images of the data set X. The padding is applied to the height and width of an image.

    :param X: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    :param pad: integer, amount of padding around each image on vertical and horizontal dimensions
    :return: padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    img_pad = np.pad(img, ((pad[0, 0], pad[0, 1]), (pad[1, 0], pad[1, 1])), 'constant')

    return img_pad


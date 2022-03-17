import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize


def get_image_rgb(origin_image):
    """

    :param origin_image: image array with n * n * 4
    :return: image array with n * n * 3
    """
    size = origin_image.shape
    res = np.zeros([size[0], size[1], 3], dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            res[i, j, 0] = origin_image[i, j, 0]
            res[i, j, 1] = origin_image[i, j, 1]
            res[i, j, 2] = origin_image[i, j, 2]
    return res


def get_image_gray(origin_image_rgb):
    """

    :param origin_image_rgb: image array with n * n * 3
    :return: image array with n * n * 1
    """
    res = np.zeros(shape=(origin_image_rgb.shape[0], origin_image_rgb.shape[1]), dtype=np.uint8)
    for i in range(origin_image_rgb.shape[0]):
        for j in range(origin_image_rgb.shape[1]):
            res[i, j] = (origin_image_rgb[i, j, 0] * 1.0 + origin_image_rgb[i, j, 1] * 1.0 + origin_image_rgb[
                i, j, 2] * 1.0) / 3
    return res


def get_histogram(img):
    histogram = [0] * 256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[int(img[i, j])] += 1
    return histogram


def get_histogram_cumule(img):
    histogram = get_histogram(img)
    histogram_cumule = [0] * 255
    histogram_cumule[0] = histogram[0]
    for i in range(1, 255):
        histogram_cumule[i] = histogram[i] + histogram_cumule[i - 1]
    return histogram_cumule


def divide_image_rpg(image_rgb):
    size = image_rgb.shape
    image_r = np.zeros([size[0], size[1]], dtype=np.uint8)
    image_g = np.zeros([size[0], size[1]], dtype=np.uint8)
    image_b = np.zeros([size[0], size[1]], dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            image_r[i, j] = image_rgb[i, j, 0]
            image_g[i, j] = image_rgb[i, j, 1]
            image_b[i, j] = image_rgb[i, j, 2]
    return image_r, image_g, image_b


def equalization(image):
    image_egalization = np.zeros(image.shape, dtype=np.uint8)
    histogram_cumule = get_histogram_cumule(image)
    n = 255
    N = image.shape[0] * image.shape[1]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            c = histogram_cumule[int(image[i, j])]
            image_egalization[i, j] = max(0, (n / N) * c - 1)
    return image_egalization


def equalization_rgb(image_rgb):
    image_r, image_b, image_g = divide_image_rpg(image_rgb)
    image_r_eq = equalization(image_r)
    image_g_eq = equalization(image_g)
    image_b_eq = equalization(image_b)
    return fusion_image_rgb(image_r_eq, image_g_eq, image_b_eq)


def fusion_image_rgb(image_r, image_g, image_b):
    size = image_r.shape
    image_rgb = np.zeros([size[0], size[1], 3], dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            image_rgb[i, j, 0] = image_r[i, j]
            image_rgb[i, j, 1] = image_g[i, j]
            image_rgb[i, j, 2] = image_b[i, j]
    return image_rgb

def otsu(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img

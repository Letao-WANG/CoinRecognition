import cv2
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
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(gray, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        # print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_img


def erosion(image_gris, elmt_struct, center):
    eros_output = np.ones((image_gris.shape[0], image_gris.shape[1]), dtype=np.uint8)

    for i in range(eros_output.shape[0]):
        for j in range(eros_output.shape[1]):
            for i2 in range(-center[0], elmt_struct.shape[0] - center[0] - 1):
                if (i + i2 < 0) or (i + i2 >= eros_output.shape[0]):
                    continue
                for j2 in range(-center[1], elmt_struct.shape[1] - center[1] - 1):
                    if j + j2 < 0 or j + j2 >= eros_output.shape[1] or elmt_struct[center[0] + i2, center[1] + j2] == 0:
                        continue

                    if image_gris[i + i2, j + j2] == 0:
                        eros_output[i, j] = 0
                        break

                if eros_output[i, j] == 0:
                    break

    return eros_output


def dilatation(image_gris, elmt_struct, center):
    eros_output = np.zeros((image_gris.shape[0], image_gris.shape[1]), dtype=np.uint8)

    for i in range(eros_output.shape[0]):
        for j in range(eros_output.shape[1]):
            for i2 in range(-center[0], elmt_struct.shape[0] - center[0] - 1):
                if i + i2 < 0 or i + i2 >= eros_output.shape[0]:
                    continue
                for j2 in range(-center[1], elmt_struct.shape[1] - center[1] - 1):
                    if j + j2 < 0 or j + j2 >= eros_output.shape[1] or elmt_struct[center[0] + i2, center[1] + j2] == 1:
                        continue

                    if image_gris[i + i2, j + j2] == 1:
                        eros_output[i, j] = 1
                        break

                if eros_output[i, j] == 1:
                    break

    return eros_output


def fermeture(image_gris, elmt_struct, center):
    img_fermeture = dilatation(image_gris, elmt_struct, center)
    img_fermeture = erosion(img_fermeture, elmt_struct, center)

    return img_fermeture


def ouverture(image_gris, elmt_struct, center):
    img_fermeture = erosion(image_gris, elmt_struct, center)

    img_fermeture = dilatation(img_fermeture, elmt_struct, center)

    return img_fermeture


def structuring_element(diameter):
    circle = np.zeros((diameter, diameter))
    center = (circle.shape[0] // 2, circle.shape[1] // 2)
    for i in range(circle.shape[0]):
        for j in range(circle.shape[1]):
            if np.sqrt((center[0] - i) ** 2 + (center[1] - j) ** 2) <= center[0]:
                circle[i, j] = 1
    return circle


def opening(image_gris, diameter):
    circle = structuring_element(diameter)
    center = (2, 2)
    image_ouverture = ouverture(image_gris, circle, center)
    return image_ouverture


def closing(image_gris, diameter):
    circle = structuring_element(diameter)
    center = (2, 2)
    image_closing = fermeture(image_gris, circle, center)
    return image_closing


def delete_background(image_gray, image):
    image_new = image
    size = image_gray.shape
    delete_top_row = 0
    delete_bottom_row = 0
    delete_left_col = 0
    delete_right_col = 0

    # delete_top_row
    for i in range(size[0]):
        delete = True
        for j in range(size[1]):
            if image_gray[i, j] != 0:
                delete = False
        if delete:
            image_new = np.delete(image_new, 0, axis=0)
        else:
            break

    # delete_bottom_row
    for i in range(size[0]-1, 0, -1):
        delete = True
        for j in range(size[1]):
            if image_gray[i, j] != 0:
                delete = False
        if delete:
            image_new = np.delete(image_new, image_new.shape[0]-1, axis=0)
        else:
            break

    # delete_left_col
    for j in range(size[1]):
        delete = True
        for i in range(size[0]):
            if image_gray[i, j] != 0:
                delete = False
        if delete:
            image_new = np.delete(image_new, 0, axis=1)
        else:
            break

    # delete_right_row
    for j in range(size[1]-1, 0, -1):
        delete = True
        for i in range(size[0]):
            if image_gray[i, j] != 0:
                delete = False
        if delete:
            image_new = np.delete(image_new, image_new.shape[1]-1, axis=1)
        else:
            break

    return image_new


def canny(img):
    t_lower = 0.7*np.average(img)  # Lower Threshold
    t_upper = 1.5*np.average(img)  # Upper threshold
    edge = cv2.Canny(img, t_lower, t_upper, L2gradient=True)
    return edge


def resize_image(image_gray):
    """
    :param image_gray:
    :return:
    """
    image = otsu(image_gray)

    diameter = 5
    image = opening(image, diameter)

    image = closing(image, diameter)

    diameter = 13
    image = closing(image, diameter)

    res = delete_background(image, image_gray)
    return res

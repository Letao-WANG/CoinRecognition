import math
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from src.util_func.util import *


def filter_sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def filter_sobel_y():
    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])


def filter_defined():
    return np.array([[2, 2, 2],
                     [-2, 0, 2],
                     [-2, -2, -2]])


def expend_image_size(origin_img):
    new_image = np.zeros(shape=(origin_img.shape[0] + 2, origin_img.shape[1] + 2))
    new_image[1:origin_img.shape[0] + 1, 1:origin_img.shape[1] + 1] = origin_img
    return new_image


def convolution_sobel(gray_img):
    gray_img = expend_image_size(gray_img)
    row = gray_img.shape[0]
    col = gray_img.shape[1]
    result = np.zeros(shape=(row, col))
    for i in range(row - 2):
        for j in range(col - 2):
            current = gray_img[i:i + 3, j:j + 3]
            multiplication_x = np.abs(sum(sum(current * filter_sobel_x())))
            multiplication_y = np.abs(sum(sum(current * filter_sobel_y())))
            result[i, j] = math.sqrt(multiplication_x ** 2 + multiplication_y ** 2)
    return result


def convolution_filter(gray_img, kernel):
    row = gray_img.shape[0]
    col = gray_img.shape[1]
    kernel_size = len(kernel)
    gray_img = expend_image_size(gray_img)
    result = np.zeros(shape=(row, col))

    for i in range(row):
        for j in range(col):
            p = int((kernel_size - 1) / 2)
            value = 0
            for a in range(-p, p):
                for b in range(-p, p):
                    value += gray_img[i-a, j-b] * kernel[a, b]
            result[i, j] = value
    return result


def main():
    # image = mpimg.imread('test_images/Bikesgray.jpg')

    image = (mpimg.imread('test_images/20cents.jpg').copy() * 255).astype(np.uint8)
    image_rgb = get_image_rgb(image)
    image_gray = get_image_gray(image_rgb)
    # image = canny(image)
    # image = resize(image, (, 1000))
    # image = convolution_filter(image, filter_sobel_x())

    image = otsu(image_gray)
    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.gray)

    diameter = 5
    image = opening(image, diameter)
    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.gray)

    image = closing(image, diameter)
    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.gray)

    diameter = 13
    image = closing(image, diameter)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)

    image = delete_background(image, image_gray)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)

    plt.show()


if __name__ == '__main__':
    main()

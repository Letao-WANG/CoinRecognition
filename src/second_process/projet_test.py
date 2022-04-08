import math
import cv2
from src.util_func.util import *
from PIL import Image
import numpy as np
from scipy.spatial import distance
np.seterr(over='ignore')


def filter_sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def filter_sobel_y():
    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])


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
    kernel_size = len(kernel)
    gray_img = expend_image_size(gray_img)
    row = gray_img.shape[0]
    col = gray_img.shape[1]

    result = np.zeros(shape=(row, col))

    for i in range(row - 2):
        for j in range(col - 2):
            current = gray_img[i:i + kernel_size, j:j + kernel_size]
            multiplication = np.abs(sum(sum(current * kernel)))
            result[i, j] = multiplication
    return result


def ahash(image_gray):
    average_image_gray = np.average(image_gray)
    new_image = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]))
    res = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]))
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            if image_gray[i, j] < average_image_gray:
                new_image[i, j] = 0
            else:
                new_image[i, j] = 1

    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            if new_image[i, j] < 1:
                res[i, j] = 0
            else:
                res[i, j] = image_gray[i, j]

    return res
    # return new_image

def main():
    # image = (mpimg.imread('test_images/20cents.jpg').copy() * 255).astype(np.uint8)
    image = mpimg.imread('test_images/Bikesgray.jpg')
    # image = (mpimg.imread('test_images/1_2.png').copy() * 255).astype(np.uint8)
    # image = get_image_rgb(image)
    # image = get_image_gray(image)
    # image_eq = equalization_rgb(image_rgb)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    # image_r, image_g, image_b = divide_image_rpg(image_rgb)
    # image_fusion = fusion_image_rgb(image_r, image_g, image_b)
    # res = resize(image, (100, 100))
    # print(image)

    # image_gray = get_image_gray(image)
    # image = resize(image, (100, 100))
    # res = equalization(image)
    image = convolution_sobel(image)
    # res = ahash(image)
    # res = convolution_sobel(res)
    # print(res)
    # res = otsu(res)
    # res = equalization(res)
    # plt.imshow(res, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


# main()


def canny(image):
    img = cv2.resize(image, (150, 150))
    t_lower = 0.7*np.average(img)  # Lower Threshold
    t_upper = 1.5*np.average(img)  # Upper threshold
    aperture_size = 5  # Aperture size

    # Applying the Canny Edge filter with L2Gradient = True
    edge = cv2.Canny(img, t_lower, t_upper, L2gradient=True)

    # cv2.imshow('original', img)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge


# image = cv2.imread("test_images/50_cents.jpg")
# image_50_cents = canny(image)


def calculate_cosine_similarity_simple(image1, image2):
    image1 = image1.flatten()
    image2 = image2.flatten()
    return np.dot(image1, image2) / (np.linalg.norm(image1, 2) * np.linalg.norm(image2, 2))


def calculate_cosine_similarity(array1, array2):
    array1 = array1.flatten()
    array2 = array2.flatten()
    dot = sum(a * b for a, b in zip(array1, array2))
    norm_a = math.sqrt(sum(a * a for a in array1))
    norm_b = math.sqrt(sum(b * b for b in array2))
    cos_sim = dot / (norm_a * norm_b)
    return cos_sim


def determine_piece(image_gray):
    image_50_cents = canny("test_images/50_cents.jpg")
    image_1_euro = canny("test_images/1_euro.jpg")

    image = canny(image_gray)
    # cv2.imshow('original', image_50_cents)
    # cv2.imshow('eg', image)
    # cv2.waitKey(0)
    cos_sim = calculate_cosine_similarity(image_50_cents, image)
    print(cos_sim)
    cos_sim = calculate_cosine_similarity(image_50_cents, image_1_euro)
    print(cos_sim)


# determine_piece("test_images/1_2.png")


def get_thum(image, size=(300, 300), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        norms.append(np.linalg.norm(vector, 2))
        # 求图片的范数
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = np.dot(a / a_norm, b / b_norm)
    return res


def test2():
    # image1 = Image.open('test_images/1_1.png')
    image1 = Image.open('test_images/50_cents.jpg')
    image2 = Image.open('test_images/1_2.png')
    cosin = image_similarity_vectors_via_numpy(image1, image2)
    print('Image cosine similarity', cosin)

# test2()


def test3():
    image = np.array([[[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                      [[0, 0, 0],
                       [0, 1, 1],
                       [1, 1, 1]],
                      [[0, 0, 0],
                       [1, 1, 1],
                       [0, 1, 1]]
                      ])

    image_gray = get_image_gray(image)
    # image_gray = np.array([[0, 0, 0],
    #                        [0, 1, 1],
    #                        [1, 1, 1]])

    # print(image_gray)
    image = delete_background(image_gray, image)
    print(image)

# test3()


def test4():
    image = (mpimg.imread('test_images/20cents.jpg').copy() * 255).astype(np.uint8)
    image_rgb = get_image_rgb(image)
    image_gray = get_image_gray(image_rgb)
    res = resize_image(image_gray)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()

test4()



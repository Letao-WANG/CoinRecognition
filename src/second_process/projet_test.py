import math
import cv2

from util_func.util import *



# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=max_val)
# plt.show()


def filter_sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def filter_sobel_y():
    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])


def expend_image_size(origin_img):
    new_image = np.zeros(shape=(origin_img.shape[0]+2, origin_img.shape[1]+2))
    new_image[1:origin_img.shape[0]+1, 1:origin_img.shape[1]+1] = origin_img
    return new_image


def convolution_sobel(gray_img):
    gray_img = expend_image_size(gray_img)
    row = gray_img.shape[0]
    col = gray_img.shape[1]

    # result = np.zeros(shape=(row, col)).astype(np.uint8)
    result = np.zeros(shape=(row, col))
    for i in range(row-2):
        for j in range(col-2):
            current = gray_img[i:i+3, j:j+3]
            multiplication_x = np.abs(sum(sum(current * filter_sobel_x())))
            multiplication_y = np.abs(sum(sum(current * filter_sobel_y())))
            result[i, j] = math.sqrt(multiplication_x**2 + multiplication_y**2)

            # result[i, j] = multiplication_x
    return result


def convolution_filter(gray_img, kernel):
    kernel_size = len(kernel)
    gray_img = expend_image_size(gray_img)
    row = gray_img.shape[0]
    col = gray_img.shape[1]

    result = np.zeros(shape=(row, col))

    for i in range(row-2):
        for j in range(col-2):
            current = gray_img[i:i+kernel_size, j:j+kernel_size]
            multiplication = np.abs(sum(sum(current * kernel)))
            result[i, j] = multiplication
    return result


def main():
    image = (mpimg.imread('test_images/1_1.png').copy() * 255).astype(np.uint8)
    # image = mpimg.imread('test_images/Bikesgray.jpg')
    # image = (mpimg.imread('test_images/1_2.png').copy() * 255).astype(np.uint8)
    image = get_image_rgb(image)
    image = get_image_gray(image)
    # image_eq = equalization_rgb(image_rgb)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    # image_r, image_g, image_b = divide_image_rpg(image_rgb)
    # image_fusion = fusion_image_rgb(image_r, image_g, image_b)
    # res = resize(image, (100, 100))
    # print(image)

    # image_gray = get_image_gray(image)
    image_gray = resize(image, (150, 150))
    res = equalization(image_gray)
    res = image_gray
    res = convolution_sobel(res)
    print(res)
    # res = otsu(res)
    # res = equalization(res)
    # plt.imshow(res, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show()


main()


def test():
    img = cv2.imread("test_images/1_2.png")  # Read image

    t_lower = 100  # Lower Threshold
    t_upper = 200  # Upper threshold
    aperture_size = 5  # Aperture size
    L2Gradient = True  # Boolean

    # Applying the Canny Edge filter with L2Gradient = True
    edge = cv2.Canny(img, t_lower, t_upper, L2gradient=L2Gradient)

    cv2.imshow('original', img)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# test()
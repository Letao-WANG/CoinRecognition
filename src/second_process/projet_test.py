import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def get_image_rgb(origin_image):
    size = origin_image.shape
    res = np.zeros([size[0], size[1], 3], dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            res[i, j, 0] = origin_image[i, j, 0]
            res[i, j, 1] = origin_image[i, j, 1]
            res[i, j, 2] = origin_image[i, j, 2]
    return res


def get_histogram(img):
    histogram = [0] * 255
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
            c = histogram_cumule[image[i, j]]
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


# amplitude = 1
# histogram = [0] * 255
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         histogram[int(img[i, j])] += 1
#
# print(histogram)
#
# histogram_cumule = [0] * 255
# histogram_cumule[0] = histogram[0]
# for i in range(1, 255):
#     histogram_cumule[i] = histogram[i] + histogram_cumule[i - 1]
#
# print(histogram_cumule)

# Egalisation

# img_egalisation = np.zeros(img.shape, dtype=np.uint8)
# n = 255
# # N = histogram_cumule[len(histogram_cumule)-1]
# N = img.shape[0] * img.shape[1]
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         c = histogram_cumule[img[i, j]]
#         img_egalisation[i, j] = max(0, (n/N) * c - 1)


# def otsu(gray):
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weight = 1.0/pixel_number
#     his, bins = np.histogram(gray, np.arange(0, 257))
#     final_thresh = -1
#     final_value = -1
#     intensity_arr = np.arange(256)
#     for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
#         pcb = np.sum(his[:t])
#         pcf = np.sum(his[t:])
#         Wb = pcb * mean_weight
#         Wf = pcf * mean_weight
#
#         mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
#         muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
#         #print mub, muf
#         value = Wb * Wf * (mub - muf) ** 2
#
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     final_img = gray.copy()
#     final_img[gray > final_thresh] = 255
#     final_img[gray < final_thresh] = 0
#     return final_img


# plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=max_val)
# plt.show()


def main():
    image = (mpimg.imread('test_images/1_1.png').copy() * 255).astype(np.uint8)
    image_rgb = get_image_rgb(image)
    # image_eq = equalization_rgb(image_rgb)

    image_r, image_g, image_b = divide_image_rpg(image_rgb)
    image_fusion = fusion_image_rgb(image_r, image_g, image_b)

    plt.imshow(image_fusion)
    plt.show()


main()

import cv2
import math
import numpy as np


def canny(path):
    img = cv2.imread(path)  # Read image
    img = cv2.resize(img, (100, 100))
    t_lower = 1*np.average(img)  # Lower Threshold
    t_upper = 1.5*np.average(img)  # Upper threshold
    edge = cv2.Canny(img, t_lower, t_upper, L2gradient=True)

    # cv2.imshow('original', img)
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge


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
    cv2.imshow('original', image_1_euro)
    cv2.imshow('eg', image)
    cv2.waitKey(0)
    cos_sim = calculate_cosine_similarity(image_50_cents, image)
    print(cos_sim)
    cos_sim = calculate_cosine_similarity(image_50_cents, image_1_euro)
    print(cos_sim)


determine_piece("test_images/1_2.png")

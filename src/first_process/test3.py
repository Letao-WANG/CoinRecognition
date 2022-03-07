import cv2
import numpy as np


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)


threshold1 = 20
threshold2 = 80
kernel = np.ones((5, 5), np.float32) / 25

img = cv2.imread(r"D:\COur\img_proj\7.jpg")

imgContour = img.copy()

imgBlur = cv2.GaussianBlur(img, (7, 7), 5)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

getContours(imgDil, imgContour)

# cv2.imshow("1 normal",img)
# cv2.imshow("2 Blur",imgBlur)
# cv2.imshow("3 GrayScale",imgGray)
# cv2.imshow("4 Canny",imgCanny)
# cv2.imshow("5 Dilatation",imgDil)
# cv2.imshow("6 Contour",imgContour)

cv2.imwrite("25.jpg", imgContour)

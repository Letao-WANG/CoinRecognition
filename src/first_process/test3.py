import cv2
import numpy as np

ratio = 0.5
decal = 0

# Vérifie si un carré est dans un autre, ne le prend pas en compte
def checkPos(t1):

    t2 = []

    for c in range(len(t1)) :
        booleen = True
        y1 =t1[c][1]
        x1 = t1[c][0]
        y1f =t1[c][3]
        x1f =t1[c][2]
        for d in range(0,len(t1)) :
            x2 = t1[d][0]
            y2 =t1[d][1]     
            x2f =t1[d][2]       
            y2f = t1[d][3]
            if x1>(x2-decal) and x1<(x2f+decal): # x du premier > au xdu second
                if x1f > (x2-decal) and x1f<(x2f+decal): # x final du premier < au x final du second
                    if y1>(y2-decal) and y1<(y2f+decal):
                        if y1f>(y2-decal) and y1f<(y2f+decal):
                            booleen = False
                            

        if booleen :
            t2.append(t1[c])
        else :
            if t1[c] in t2:
                t2.remove(t1[c])
    return t2

            
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 7000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            plusGrand = w
            plusPetit = h
            if h> w:
                plusGrand = h
                plusPetit = w
            if plusPetit/plusGrand >ratio :
                t.append((x,y,x + w, y + h))
            ind+=1

    t3 = checkPos(t)
    for c in range(len(t3)):
        cv2.rectangle(imgContour, (t3[c][0],t3[c][1]),(t3[c][2],t3[c][3]) , (0, 255, 0), 5)
        #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)


# Load image
def check_empty_img(arg):
    img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpeg")
    if img is None:
        img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpg")
    elif img is None:
        img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".png")
    return img

for a in range(0, 60,2):
    threshold1 = 50 #20
    threshold2 = 60 #80
    kernel = np.ones((5, 5), np.float32) / 25

    img = check_empty_img(a)
    if img is None :
        continue
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

    cv2.imwrite(str(a)+".jpg", imgContour)



import cv2
import numpy as np
import json

ratio = 0.5


# Retourne l'ensemble des rectangles qui ne sont pas inclut dans un autre
def checkSquareIn(t1):
    t2 = []
    # Pour chaque éléments de t1
    for c in range(len(t1)):
        booleen = True
        # On récupére les coordonnées 
        y1 = t1[c][1]
        x1 = t1[c][0]
        y1f = t1[c][3]
        x1f = t1[c][2]
        # Pour chaque éléments de t1
        for d in range(0, len(t1)):
            # On récupére les coordonnées 
            x2 = t1[d][0]
            y2 = t1[d][1]
            x2f = t1[d][2]
            y2f = t1[d][3]
            # Vérification d'inclusion 
            if x1 > x2 and x1 < x2f:  # x du premier > au xdu second
                if x1f > x2 and x1f < x2f:  # x final du premier < au x final du second
                    if y1 > y2 and y1 < y2f:
                        if y1f > y2 and y1f < y2f:
                            booleen = False

        if booleen:
            t2.append(t1[c])
        else:
            if t1[c] in t2:
                t2.remove(t1[c])
    return t2


# Vérifie si deux rectangles ont une intersection, si plus de 50%
# est partagé, les réunis, sinon ne fais rien 
def SquareIntersection(t1):
    t2 = []
    t4 = []
    tcopy = []
    for l in range(0, len(t1)):
        tcopy.append(t1[l])

    doisRefaire = True

    while (doisRefaire):
        doisRefaire = False
        t2 = []
        for c in range(0, len(tcopy)):
            if doisRefaire:
                break
            booleen = False
            y1 = tcopy[c][1]
            x1 = tcopy[c][0]
            y1f = tcopy[c][3]
            x1f = tcopy[c][2]
            estModifie = False
            lenn = len(tcopy)
            for d in range(0, lenn):
                x2 = tcopy[d][0]
                y2 = tcopy[d][1]
                x2f = tcopy[d][2]
                y2f = tcopy[d][3]

                booleen = False
                # En haut à gauche
                if x2 > x1 and x2 < x1f and y2 > y1 and y2 < y1f:
                    booleen = True
                    if x2f > x1f:
                        longueur = x1f - x2
                    else:
                        longueur = x2f - x2

                    if y2f > y1:
                        hauteur = y1f - y2
                    else:
                        hauteur = y2f - y2
                # En haut à droite
                elif x2f > x1 and x2f < x1f and y2 > y1 and y2 < y1f:
                    booleen = True
                    if x2 < x1:
                        longueur = x2f - x1
                    else:
                        longueur = x2f - x2

                    if y2f > y1:
                        hauteur = y1f - y2
                    else:
                        hauteur = y2f - y2
                # En bas à gauche
                elif x2 > x1 and x2 < x1f and y2f > y1 and y2f < y1f:
                    booleen = True
                    if x2f > x1f:
                        longueur = x1f - x2
                    else:
                        longueur = x2f - x2

                    if y2 < y1:
                        hauteur = y2f - y1
                    else:
                        hauteur = y2f - y2
                # En bas a droite
                elif x2f > x1 and x2f < x1f and y2f > y1 and y2f < y1f:
                    booleen = True
                    if x2 < x1:
                        longueur = x2f - x1
                    else:
                        longueur = x2f - x2

                    if y2 < y1:
                        hauteur = y2f - y1
                    else:
                        hauteur = y2f - y2
                if booleen:
                    aire = hauteur * longueur
                    aire1 = (x1f - x1) * (y1f - y1)
                    aire2 = (x2f - x2) * (y2f - y2)
                    ratio1 = aire / aire1
                    ratio2 = aire / aire2
                    if ratio1 >= 0.4 or ratio2 >= 0.4:
                        newSquare = CombineSquare(tcopy[c], tcopy[d])
                        estModifie = True
                        tcopy.pop(d)
                        if (d < c):
                            ind = c - 1
                        else:
                            ind = c
                        tcopy.pop(ind)
                        tcopy.append(newSquare)
                        c = lenn + 1
                        d = lenn + 1
                        doisRefaire = True
                        break

            if not (estModifie):
                t2.append(tcopy[c])

    return t2


# Combine deux carrés, retourne un tuple de la forme (x,y,xf,yf).
def CombineSquare(square1, square2):
    x1 = square1[0]
    x1f = square1[2]
    y1 = square1[1]
    y1f = square1[3]

    x2 = square2[0]
    x2f = square2[2]
    y2 = square2[1]
    y2f = square2[3]

    if x1 < x2:
        x = x1
    else:
        x = x2
    if x1f > x2f:
        xf = x1f
    else:
        xf = x2f
    if y1 < y2:
        y = y1
    else:
        y = y2
    if y1f > y2f:
        yf = y1f
    else:
        yf = y2f

    return (x, y, xf, yf)


# Affiche les rectangles
def getContours(img, imgContour, imgNormal):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000:  # 7000
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            plusGrand = w
            plusPetit = h
            if h > w:
                plusGrand = h
                plusPetit = w
            if plusPetit / plusGrand > ratio:
                t.append((x, y, x + w, y + h))
            ind += 1

    # t3 = checkSquareIn(t)
    t3 = SquareIntersection(t)
    images = []
    for c in range(len(t3)):
        cv2.rectangle(imgContour, (t3[c][0], t3[c][1]), (t3[c][2], t3[c][3]), (0, 255, 0), 5)
        images.append(imgNormal[t3[c][1]:t3[c][3], t3[c][0]:t3[c][2]])
        # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return images


# Charge l'image arg
def check_empty_img(arg):
    img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpeg")
    if img is None:
        img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpg")
    if img is None:
        img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".png")
    return img


# Méthode principal
def main():
    # Le nombre de piece bien detecter (le nombre de piece sur l'image correspond bien)
    sommeSuccess = 0

    # Pour chaque image
    for a in range(0, 60, 2):
        threshold1 = 50  # 20
        threshold2 = 60  # 80
        kernel = np.ones((5, 5), np.float32) / 25

        img = check_empty_img(a)
        if img is None:
            continue
        imgContour = img.copy()

        imgBlur = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        images = getContours(imgDil, imgContour, img)
        if (len(images) == 0):
            imgBlur = cv2.GaussianBlur(img, (7, 7), 5)
            # imgBlur = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

            images = getContours(imgDil, imgContour, img)

        with open(r"D:\COur\img_proj\\" + str(a) + ".json") as json_data:
            data_dict = json.load(json_data)
            if (len(data_dict["shapes"]) == len(images)):
                print("image " + str(a) + " : Success")
                sommeSuccess += 1
            else:
                print("image " + str(a) + " : Failure")

        show = False
        if (show):
            for i in range(len(images)):
                cv2.imshow("image", images[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # cv2.imshow("1 normal",img)
        # cv2.imshow("2 Blur",imgBlur)
        # cv2.imshow("3 GrayScale",imgGray)
        # cv2.imshow("4 Canny",imgCanny)
        # cv2.imshow("5 Dilatation",imgDil)
        # cv2.imshow("6 Contour",imgContour)

        cv2.imwrite(str(a) + ".jpg", imgContour)

    print(str(sommeSuccess) + " Success")


main()

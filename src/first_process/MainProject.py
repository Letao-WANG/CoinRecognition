import cv2
import numpy as np
import json

# import os

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
            if x1 > x2 and x1 < x2f:  # x du premier est compris entre x du sencond et x final du second
                if x1f > x2 and x1f < x2f:  # x final du premier est compris entre x du sencond et x final du second
                    if y1 > y2 and y1 < y2f:  # y du premier est compris entre y du sencond et y final du second
                        if y1f > y2 and y1f < y2f:  # y final du premier est compris entre y du sencond et y final du second
                            booleen = False

        if booleen:
            t2.append(t1[c])
        else:
            if t1[c] in t2:
                t2.remove(t1[c])
    return t2


# Vérifie si deux rectangles ont une intersection, si plus de 40%
# est partagé, les réunis, sinon ne fais rien 
def SquareIntersection(t1):
    t2 = []
    tcopy = []
    # copie du tableau dans tcopy
    for l in range(0, len(t1)):
        tcopy.append(t1[l])
    # si on a fréunis deux rectangles on revérifies pour tous les rectangles
    doisRefaire = True

    while (doisRefaire):
        doisRefaire = False
        t2 = []
        # pour chaque rectangle
        for c in range(0, len(tcopy)):
            if doisRefaire:
                break
            # On récupére les coordonnées 
            y1 = tcopy[c][1]
            x1 = tcopy[c][0]
            y1f = tcopy[c][3]
            x1f = tcopy[c][2]
            estModifie = False
            lenn = len(tcopy)
            # pour chaque rectangle
            for d in range(0, lenn):
                # On récupére les coordonnées 
                x2 = tcopy[d][0]
                y2 = tcopy[d][1]
                x2f = tcopy[d][2]
                y2f = tcopy[d][3]

                intersect = False
                # Vérification d'intersection
                # En haut à gauche
                if x2 > x1 and x2 < x1f and y2 > y1 and y2 < y1f:
                    intersect = True
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
                    intersect = True
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
                    intersect = True
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
                    intersect = True
                    if x2 < x1:
                        longueur = x2f - x1
                    else:
                        longueur = x2f - x2

                    if y2 < y1:
                        hauteur = y2f - y1
                    else:
                        hauteur = y2f - y2
                # S'il y a intersection
                if intersect:
                    # calcul de l'aire dintersection
                    aire = hauteur * longueur
                    # calcul de l'aire du premier rectangle
                    aire1 = (x1f - x1) * (y1f - y1)
                    # calcul de l'aire du second rectangle
                    aire2 = (x2f - x2) * (y2f - y2)
                    # calcul des ratios
                    ratio1 = aire / aire1
                    ratio2 = aire / aire2
                    if ratio1 >= 0.4 or ratio2 >= 0.4:
                        # Combine les rectangles
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


# Combine deux rectangles, retourne un tuple de la forme (x,y,xf,yf).
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


# permet de récupérer les pieces des images.      
def getPieces(img, imgContour, imgNormal):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    # Pour chaque contour
    for cnt in contours:
        # calculer l'air
        area = cv2.contourArea(cnt)
        # Si l'air est suffisement grande
        if area > 8000:  # 7000
            # Déssiner les contours
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            # verification du ration W/H pour etre sur de ne pas prendre des rectangles trop fin
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
        # Dessiner les rectangles
        cv2.rectangle(imgContour, (t3[c][0], t3[c][1]), (t3[c][2], t3[c][3]), (0, 255, 0), 5)
        # Découper les images par piece
        images.append(imgNormal[t3[c][1]:t3[c][3], t3[c][0]:t3[c][2]])
        # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return images


# Charge l'image arg
def check_empty_img(arg):
    img = cv2.imread(r"origin_images\\" + str(arg) + ".jpeg")
    if img is None:
        img = cv2.imread(r"origin_images\\" + str(arg) + ".jpg")
    if img is None:
        img = cv2.imread(r"origin_images\\" + str(arg) + ".png")
    return img


# Méthode principal
def main():
    # Le nombre de piece bien detecter (le nombre de piece sur l'image correspond bien)
    sommeSuccess = 0

    # Pour chaque image
    for a in range(0, 60, 2):

        # Récupere l'image
        img = check_empty_img(a)
        if img is None:
            continue

        kernel = np.ones((5, 5), np.float32) / 25
        imgContour = img.copy()
        imgBlur = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)  # flou
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)  # passage en gris

        # v = np.median( imgGray )
        # sigma = 0.9
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))

        threshold1 = 50  # 20 50 60
        threshold2 = 60  # 80 60 100

        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)  # Canny pour contour
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)  # Dilatation pour augmenter les contours

        images = getPieces(imgDil, imgContour, img)

        # utilisation d'un flou différent (permet d'obtenir d'autre piece)
        if (len(images) == 0):
            imgBlur = cv2.GaussianBlur(img, (7, 7), 5)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

            images = getPieces(imgDil, imgContour, img)

        # Vérifications des résultats obtenus
        with open(r"origin_images\\" + str(a) + ".json") as json_data:
            data_dict = json.load(json_data)
            if (len(data_dict["shapes"]) == len(images)):
                print("image " + str(a) + " : Success")
                sommeSuccess += 1
            else:
                print("image " + str(a) + " : Failure")

        show = False
        saveImgContour = False
        save = False
        saveOther = False
        # Affichage
        if (show):
            for i in range(len(images)):
                cv2.imshow("image", images[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cv2.imshow("1 normal", img)
            cv2.imshow("2 Blur", imgBlur)
            cv2.imshow("3 GrayScale", imgGray)
            cv2.imshow("4 Canny", imgCanny)
            cv2.imshow("5 Dilatation", imgDil)
            cv2.imshow("6 Contour", imgContour)

        # Sauvegarde
        if (save):
            # os.mkdir("Result\\"+str(a))
            for i in range(len(images)):
                cv2.imwrite("Result\\" + str(a) + "\piece" + str(i) + ".jpg", images[i])

        if (saveImgContour):
            cv2.imwrite(str(a) + ".jpg", imgContour)

        if (saveOther):
            cv2.imwrite(str(a) + "blur.jpg", imgBlur)
            cv2.imwrite(str(a) + "gray.jpg", imgGray)
            cv2.imwrite(str(a) + "canny.jpg", imgCanny)
            cv2.imwrite(str(a) + "dilatation.jpg", imgDil)
            cv2.imwrite(str(a) + "contour.jpg", imgContour)

    print(str(sommeSuccess) + " Success")


main()

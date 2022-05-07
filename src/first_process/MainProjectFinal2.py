import cv2
import numpy as np
import json
import math
import os
import matplotlib.pyplot as plt
import copy
# from google.colab.patches import cv2.imshow

import warnings

warnings.simplefilter("error", np.VisibleDeprecationWarning)

#import imagehash
from PIL import Image,ImageChops
from skimage.metrics import structural_similarity as ssim

ratio = 0.5

# La quantité de pièce similaire à celles contenue dans les fichiers .json
counter_coin = 0
# Contient tous les points du fichiers .json 
array_point = np.empty((0,2))

# quantité de pièce
total_coin = 0

# Variable contenant les pièces
coin_liste = np.zeros(20)

# Variable contenant la valeur des pièces
coin_value_liste = np.zeros(20)
coin_value_liste = coin_value_liste.astype('str')

# Variable contenant les pièces, pris du .json
coin_liste_json = np.empty((0,2))

# Variable contenant la valeur des pièces pris du .json
coin_value_liste_json = np.empty((0,2))
coin_value_liste_json = coin_value_liste_json.astype('str')

# Retourne l'ensemble des rectangles qui ne sont pas inclut dans un autre
def checkSquareIn(t1):

    t2 = []
    # Pour chaque éléments de t1
    for c in range(len(t1)) :
        booleen = True
        # On récupére les coordonnées 
        y1 =t1[c][1]
        x1 = t1[c][0]
        y1f =t1[c][3]
        x1f =t1[c][2]
        #Pour chaque éléments de t1
        for d in range(0,len(t1)) :
            # On récupére les coordonnées 
            x2 = t1[d][0]
            y2 =t1[d][1]     
            x2f =t1[d][2]       
            y2f = t1[d][3]
            # Vérification d'inclusion 
            if x1>x2 and x1<x2f: # x du premier est compris entre x du sencond et x final du second
                if x1f > x2 and x1f<x2f: # x final du premier est compris entre x du sencond et x final du second
                    if y1>y2 and y1<y2f: # y du premier est compris entre y du sencond et y final du second
                        if y1f>y2 and y1f<y2f: # y final du premier est compris entre y du sencond et y final du second
                            booleen = False
                            

        if booleen :
            t2.append(t1[c])
        else :
            if t1[c] in t2:
                t2.remove(t1[c])
    return t2

# Vérifie si deux rectangles ont une intersection, si plus de 40%
# est partagé, les réunis, sinon ne fais rien 
def SquareIntersection(t1):
    t2 = []
    tcopy = []
    # copie du tableau dans tcopy
    for l in range(0,len(t1)) :
        tcopy.append(t1[l])
    # si on a fréunis deux rectangles on revérifies pour tous les rectangles
    doisRefaire = True

    while(doisRefaire):
        doisRefaire = False
        t2 = []
        # pour chaque rectangle
        for c in range(0,len(tcopy)) :
            if doisRefaire:
                break
            # On récupére les coordonnées 
            y1 =tcopy[c][1]
            x1 = tcopy[c][0]
            y1f =tcopy[c][3]
            x1f =tcopy[c][2]
            estModifie = False
            lenn = len(tcopy)
            # pour chaque rectangle
            for d in range(0,lenn) :
                # On récupére les coordonnées 
                x2 = tcopy[d][0]
                y2 =tcopy[d][1]     
                x2f =tcopy[d][2]       
                y2f = tcopy[d][3]
                  
                intersect = False
                #Vérification d'intersection
                #En haut à gauche
                if x2>x1 and x2<x1f and y2>y1  and y2<y1f:
                    intersect= True
                    if x2f>x1f:
                        longueur = x1f - x2
                    else:
                        longueur = x2f -x2     

                    if y2f>y1:
                        hauteur = y1f-y2
                    else:
                        hauteur = y2f-y2
                #En haut à droite
                elif x2f>x1 and x2f<x1f and y2>y1  and y2<y1f:
                    intersect = True
                    if x2<x1:
                        longueur = x2f - x1
                    else:
                        longueur = x2f -x2     

                    if y2f>y1:
                        hauteur = y1f-y2
                    else:
                        hauteur = y2f-y2
                #En bas à gauche
                elif x2>x1 and x2<x1f and y2f>y1  and y2f<y1f:
                    intersect = True
                    if x2f>x1f:
                        longueur = x1f - x2
                    else:
                        longueur = x2f -x2     

                    if y2<y1:
                        hauteur = y2f-y1
                    else:
                        hauteur = y2f-y2
                #En bas a droite
                elif x2f>x1 and x2f<x1f and y2f>y1  and y2f<y1f:
                    intersect = True
                    if x2<x1:
                        longueur = x2f - x1
                    else:
                        longueur = x2f -x2    

                    if y2<y1:
                        hauteur = y2f-y1
                    else:
                        hauteur = y2f-y2
                #S'il y a intersection
                if intersect:
                    # calcul de l'aire dintersection
                    aire = hauteur * longueur
                    # calcul de l'aire du premier rectangle
                    aire1 = (x1f-x1)*(y1f-y1)
                    # calcul de l'aire du second rectangle
                    aire2 = (x2f-x2)*(y2f-y2)
                    # calcul des ratios
                    ratio1 = aire/aire1
                    ratio2 = aire/aire2
                    if ratio1>=0.4 or ratio2 >= 0.4:
                        #Combine les rectangles
                        newSquare = CombineSquare(tcopy[c],tcopy[d])
                        estModifie = True
                        tcopy.pop(d)
                        if(d<c):
                            ind = c-1
                        else:
                            ind = c
                        tcopy.pop(ind)
                        tcopy.append(newSquare)
                        c = lenn+1
                        d = lenn+1
                        doisRefaire = True
                        break

        
            if not(estModifie):
                t2.append(tcopy[c])

    return t2


    
#Combine deux rectangles, retourne un tuple de la forme (x,y,xf,yf).
def CombineSquare(square1,square2):
    x1 = square1[0]
    x1f = square1[2]
    y1 = square1[1]
    y1f = square1[3]

    x2 = square2[0]
    x2f = square2[2]
    y2 = square2[1]
    y2f = square2[3]

    if x1<x2 :
        x = x1
    else:
        x = x2
    if x1f>x2f :
        xf = x1f
    else:
        xf = x2f
    if y1<y2 :
        y = y1
    else:
        y = y2
    if y1f>y2f :
        yf = y1f
    else:
        yf = y2f

    return (x,y,xf,yf)
    
    
# permet de récupérer les pieces des images.      
def getPieces(img, imgContour,imgNormal):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    # Pour chaque contour
    for cnt in contours:
        # calculer l'air
        area = cv2.contourArea(cnt)
        # Si l'air est suffisement grande
        if area > 8000: #7000
            # Déssiner les contours
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            #verification du ration W/H pour etre sur de ne pas prendre des rectangles trop fin
            plusGrand = w
            plusPetit = h
            if h> w:
                plusGrand = h
                plusPetit = w
            if plusPetit/plusGrand >ratio :
                t.append((x,y,x + w, y + h))
            ind+=1

    #t3 = checkSquareIn(t)
    t3 = SquareIntersection(t)
    images = []
    for c in range(len(t3)):
        #Dessiner les rectangles
        cv2.rectangle(imgContour, (t3[c][0],t3[c][1]),(t3[c][2],t3[c][3]) , (0, 255, 0), 5)
        #Découper les images par piece
        images.append(imgNormal[t3[c][1]:t3[c][3],t3[c][0]:t3[c][2]] )
        #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return images

# permet de récupérer les pieces des images.      
def getPieces2(img, imgContour,imgNormal, img_num, first):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    # Pour chaque contour
    for cnt in contours:
        # calculer l'air
        area = cv2.contourArea(cnt)
        # Si l'air est suffisement grande
        if area > 8000: #7000
            # Déssiner les contours
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            #verification du ration W/H pour etre sur de ne pas prendre des rectangles trop fin
            plusGrand = w
            plusPetit = h
            if h> w:
                plusGrand = h
                plusPetit = w
            if plusPetit/plusGrand >ratio :
                t.append((x,y,x + w, y + h))
            ind+=1
    listeTmp = jsontest(img_num, first)
    #t3 = checkSquareIn(t)
    t3 = SquareIntersection(t)
    images = []
    for c in range(len(t3)):
        #Dessiner les rectangles
        cv2.rectangle(imgContour, (t3[c][0],t3[c][1]),(t3[c][2],t3[c][3]) , (0, 255, 0), 5)

        point_in_circle(listeTmp, t3, c)

        global array_point
        if(len(array_point) != 0):
            tmp = point_in_circle2(array_point, t3, c)
            array_point = point_in_circle2(array_point, t3, c)

        
        #Découper les images par piece
        images.append(imgNormal[t3[c][1]:t3[c][3],t3[c][0]:t3[c][2]] )
        #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return images

def quotient(val1, val2):
    return int(((val2-val1)/100) * 20)

# Vérifie si les deux points sont compris dans l'image
def point_in_circle(listeTmp, t3, c):
    if listeTmp is None :
      return
    for coordoX in range(0,len(listeTmp),2):
        tmpx = quotient(t3[c][0], t3[c][2])
        tmpy = quotient(t3[c][1], t3[c][3])
        if(listeTmp[coordoX][0] >= t3[c][0]-tmpx and listeTmp[coordoX][0] <= t3[c][2]+tmpx
           and listeTmp[coordoX][1] >= t3[c][1]-tmpy and listeTmp[coordoX][1] <= t3[c][3]+tmpy ):
            #for coordoY in range(0, len(listeTmp[coordoX])):
            if(listeTmp[coordoX+1][0] >= t3[c][0]-tmpx and listeTmp[coordoX+1][0] <= t3[c][2]+tmpx
                and listeTmp[coordoX+1][1] >= t3[c][1]-tmpy and listeTmp[coordoX+1][1] <= t3[c][3]+tmpy ):

                array1 = np.array([listeTmp[coordoX][0], listeTmp[coordoX][1]])
                array12 = np.array([listeTmp[coordoX+1][0], listeTmp[coordoX+1][1]])
                array2 = np.array([t3[c][0], t3[c][1]])
                array22 = np.array([t3[c][2]-((t3[c][2]-t3[c][0])/2), t3[c][1]])
                surface1 = circle_surface(array1, array12)
                surface2 = circle_surface(array2, array22)

                if(compare_circle_surface(surface1, surface2)):
                    global counter_coin, coin_liste
                    counter_coin += 1
                    coin_liste[c] = listeTmp[coordoX][0]
                
# Vérifie si au moins 70 des points sont compris dans l'image
def point_in_circle2(array_point, t3, c):
    quantite = 0
    print('taille : '+str(len(array_point)))
    for coordoX in range(0,len(array_point)):
        tmpx = quotient(t3[c][0], t3[c][2])
        tmpy = quotient(t3[c][1], t3[c][3])
        if(array_point[coordoX][0] >= t3[c][0]-tmpx and array_point[coordoX][0] <= t3[c][2]+tmpx
           and array_point[coordoX][1] >= t3[c][1]-tmpy and array_point[coordoX][1] <= t3[c][3]+tmpy ):
            quantite += 1
    print('quantite : '+str(quantite))
    if(quantite/len(array_point) >= 0.75):
        global counter_coin, coin_liste
        counter_coin += 1
        coin_liste[c] = array_point[0][0]
        print('\nligne 359 : '+str(coin_liste))
    #array_point = np.empty((0,2))
    return array_point

# Calcul la surface d'un cercle
def circle_surface(point1, point2):
    rayon = np.linalg.norm(point1 - point2)
    return np.pi *(rayon*rayon)

# Compare la surface du premier cercle avec le second
# s'ils ont une surface similaire return True, sinon return False
def compare_circle_surface(surface1, surface2):
    if surface1 / surface2 >= 0.45 and surface1 / surface2 <1.15 :
        return True
    else :
        return False

# Récuppère les informations nécessaire dans un fichier .json
def jsontest(img_num, first):
    
    #for img_num in range(0,nb_img):
    path = str(img_num) +".json"
    #Vérifie si l'image existe
    try:
        with open(path) as fichier_data:
            data = json.load(fichier_data)
        first_time = True
        another_first_time = True
        circle = np.empty((0,2))
        if(first):
            print ('----------'+data['imagePath']+'----------\n')
            
            global total_coin
            total_coin += len(data['shapes'])
            print('il y a : ' +str(len(data['shapes']))+' pièces\n')
            tmp = np.zeros(20)
            tmp = tmp.astype('str')
            
            global coin_liste
            #coin_liste = np.empty((0,2))
            #print(tmp)
            
            #coin_liste = np.insert(coin_liste,img_num, tmp)
            coin_liste = np.zeros(20)
            coin_liste = coin_liste.astype('str')

            global coin_value_liste
            #coin_value_liste = np.insert(coin_value_liste,img_num, tmp)
            coin_value_liste = np.zeros(20)
            coin_value_liste = coin_value_liste.astype('str')

            global coin_value_liste_json, coin_liste_json
            for i in range(0, len(data['shapes'])):
                coin_value_liste_json = np.insert(coin_value_liste_json, i, data['shapes'][i]['label'])
                coin_liste_json = np.insert(coin_liste_json, i, data['shapes'][i]['points'][0][0])
                
        for allShapes in range(0,len(data['shapes'])):
            #Pour tous les points utilisés lors du détourage
            #Deux points si ça a été fait avec un cercle (outil), les deux points forment le rayon,
            #plus si il y a pas eu l'utilisation de l'outil pour former un cercle
            for i in range(0,len(data['shapes'][allShapes]['points']), 2):

                if(len(data['shapes'][allShapes]['points']) == 2):

                    circle = np.vstack([circle, data['shapes'][allShapes]['points']])
                else:
                    #Dans le cas ou il y a plus de deux points
                    if(another_first_time):
                        global array_point
                        array_point = np.vstack([array_point, data['shapes'][allShapes]['points']])
                        another_first_time = False
                    else:
                        continue
        return circle
    except IOError:
        print("le fichier n'existe pas")
        return None


def calculate_cosine_similarity(array1, array2):
    array1 = array1.flatten()
    array2 = array2.flatten()
    dot = sum(int(a) * int(b) for a, b in zip(array1, array2))
    norm_a = math.sqrt(sum(int(a) * int(a) for a in array1))
    norm_b = math.sqrt(sum(int(b) * int(b) for b in array2))
    cos_sim = dot / (norm_a * norm_b)
    return cos_sim

# Determine la valeur en fonction de l'histo
def determine_piece2(image):
    IMG_SIZE = (int(image.shape[0]/2),int(image.shape[1]/2))

    target_img = cv2.resize(image, IMG_SIZE)
    target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])
    cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)

    comparing_img = check_empty_img("1_ctm")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret1 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret1)

    comparing_img = check_empty_img("2_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret2 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret2)

    comparing_img = check_empty_img("5_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret3 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret3)

    comparing_img = check_empty_img("10_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret4 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret4)

    comparing_img = check_empty_img("20_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret5 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret5)

    comparing_img = check_empty_img("50_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret6 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret6)

    comparing_img = check_empty_img("1_euro")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)

    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret7 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret7)

    comparing_img = check_empty_img("2_euros")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret8 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret8)

    m = max(ret1,ret2,ret3,ret4,ret5,ret6,ret7,ret8)
    print('max : '+str(m))

    if(m == ret1):
        print("1 cent")
        return "Piece1ctm"
    if(m == ret2):
        print("2 cents")
        return "Piece2ctms"
    if(m == ret3):
        print("5 cents")
        return "Piece5ctms"
    if(m == ret4):
        print("10 cents")
        return "Piece10ctms"
    if(m == ret5):
        print("20 cents")
        return "Piece20ctms"
    if(m == ret6):
        print("50 cents")
        return "Piece50ctms"
    if(m == ret7):
        print("1 euro")
        return "Piece1euro"
    if(m == ret8):
        print("2 euros")
        return "Piece2euros"

# Determine la valeur en fonction du cosine similarity
def determine_piece(deter_image):
    if isinstance(deter_image, type(None)):
      print ("problem")
      return "1_ctm"
    assert not isinstance(deter_image,type(None)), 'image not found'
    image_gray = get_image_gray(deter_image)
    coin_names = ["1_ctm", "2_ctms", "5_ctms", "10_ctms", "20_ctms", "50_ctms", "1_euro", "2_euros"]
    coin_res = ["Piece1ctm", "Piece2ctms", "Piece5ctms", "Piece10ctms", "Piece20ctms", "Piece50ctms", "Piece1euro", "Piece2euros"]
    image = cut_coin_image(image_gray)

    images_standard = []
    for coin in coin_names:
        image_standard = check_empty_img(coin)
        image_standard_gray = get_image_gray(image_standard)
        image_cut = cut_coin_image(image_standard_gray)
        images_standard.append(image_cut)

    max_cos = 0
    cpt = 0
    index = 0
    for image_standard in images_standard:
        cv2.imshow(image_standard)
        cv2.imshow(image)
        cos = calculate_cosine_similarity(image_standard, image)
        print("cosine similarity: " + str(cos))
        if cos > max_cos:
            max_cos = cos
            index = cpt
        cpt += 1

    print(coin_res[index])
    return coin_res[index]
    
# Determine la valeur de la piece en fonction du hash
# def determine_piece3(image):
#
#     IMG_SIZE = (200, 200)
#
#     image_1_cent = check_empty_img("test_images\\1_ctm")
#     image_2_cents = check_empty_img("test_images\\2_ctms")
#     image_5_cents = check_empty_img("test_images\\5_ctms")
#     image_10_cents = check_empty_img("test_images\\10_ctms")
#     image_20_cents = check_empty_img("test_images\\20_ctms")
#     image_50_cents = check_empty_img("test_images\\50_ctms")
#     image_1_euro = check_empty_img("test_images\\1_euro")
#     image_2_euros = check_empty_img("test_images\\2_euros")
#
#     image_1_cent = cv2.resize(image_1_cent, IMG_SIZE)
#     image_2_cents = cv2.resize(image_2_cents, IMG_SIZE)
#     image_5_cents = cv2.resize(image_5_cents, IMG_SIZE)
#     image_10_cents = cv2.resize(image_10_cents, IMG_SIZE)
#     image_20_cents = cv2.resize(image_20_cents, IMG_SIZE)
#     image_50_cents = cv2.resize(image_50_cents, IMG_SIZE)
#     image_1_euro = cv2.resize( image_1_euro, IMG_SIZE)
#     image_2_euros = cv2.resize(image_2_euros, IMG_SIZE)
#
#
#     hash0 = imagehash.average_hash(Image.open("1_ctm_bis"))
#     hash1 = imagehash.average_hash(Image.open("test_images\\2_ctms"))
#     cutoff = 5  # maximum bits that could be different between the hashes.
#
#     if hash0 - hash1 < cutoff:
#       print('images are similar')
#     else:
#       print('images are not similar')

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


def erosion_verify_element(content, element):
    for i in range(content.shape[0]):
        for j in range(content.shape[1]):
            if element[i, j] == 1:
                if not content[i, j] == 1:
                    return False
    return True


def dilation_verify_element(content, element):
    for i in range(content.shape[0]):
        for j in range(content.shape[1]):
            if element[i, j] == 1:
                if content[i, j] == 1:
                    return True
    return False


def erosion(image, element):
    size = image.shape
    size_element = element.shape
    img = np.zeros([size[0], size[1]], dtype=bool)
    for i in range(size_element[0], size[0]-size_element[0]):
        for j in range(size_element[1], size[1]-size_element[1]):
            content = image[i:i+size_element[0], j:j+size_element[1]]
            is_verify = erosion_verify_element(content, element)
            if is_verify:
                img[i+size_element[0]//2, j+size_element[1]//2] = 1
    return img


def dilation(image, element):
    size = image.shape
    size_element = element.shape
    img = np.zeros([size[0], size[1]], dtype=bool)
    for i in range(size_element[0], size[0]-size_element[0]):
        for j in range(size_element[1], size[1]-size_element[1]):
            content = image[i:i+size_element[0], j:j+size_element[1]]
            is_verify = dilation_verify_element(content, element)
            if is_verify:
                img[i+size_element[0]//2, j+size_element[1]//2] = 1
    return img


def opening(image, element):
    image_erosion = erosion(image, element)
    image_opening = dilation(image_erosion, element)
    return image_opening


def closing(image, element):
    image_dilatation = dilation(image, element)
    image_closing = erosion(image_dilatation, element)
    return image_closing


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
    t_lower = 0.1*np.average(img)  # Lower Threshold
    t_upper = 0.6*np.average(img)  # Upper threshold
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


def reduce_size(image, row):
    scale = image.shape[0]/row
    col = int(image.shape[1]/scale)
    return cv2.resize(image, (col, row))


def thresholding(image, threshold):
    img = np.zeros(image.shape[0], image.shape[1], dtype=bool)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < threshold:
                img = 0
            else:
                img = 1
    return img


def cut_coin_image(image_gray, modify_size=250, res_size=200):
    """
    Remove redundant background of coins
    :param image_gray: image gris
    :param modify_size: first reduce size of image
    :param res_size: size of result image
    :return: image gris
    """
    image_gray = reduce_size(image_gray, modify_size)

    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    ret, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    diameter = 5
    image = opening(image, diameter)
    image = delete_background(image, image_gray)

    image = reduce_size(image, res_size)
    return image

def mse(imageA,imageB):
    

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
	
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# Determine la valeur en fonction du ssim
def compare_images(image):
    IMG_SIZE = (int(image.shape[0]*0.5), int(image.shape[1]*0.5))

    imageA = cv2.resize(image, IMG_SIZE)
    image_1_cent = check_empty_img("test_images\\1_ctm")
    image_2_cents = check_empty_img("test_images\\2_ctms")
    image_5_cents = check_empty_img("test_images\\5_ctms")
    image_10_cents = check_empty_img("test_images\\10_ctms")
    image_20_cents = check_empty_img("test_images\\20_ctms")
    image_50_cents = check_empty_img("test_images\\50_ctms")
    image_1_euro = check_empty_img("test_images\\1_euro")
    image_2_euros = check_empty_img("test_images\\2_euros")

    image_1_cent = cv2.resize(image_1_cent, IMG_SIZE)
    image_2_cents = cv2.resize(image_2_cents, IMG_SIZE)
    image_5_cents = cv2.resize(image_5_cents, IMG_SIZE)
    image_10_cents = cv2.resize(image_10_cents, IMG_SIZE)
    image_20_cents = cv2.resize(image_20_cents, IMG_SIZE)
    image_50_cents = cv2.resize(image_50_cents, IMG_SIZE)
    image_1_euro = cv2.resize( image_1_euro, IMG_SIZE)
    image_2_euros = cv2.resize(image_2_euros, IMG_SIZE)
    
    # compute the mean squared error and structural similarity
    # index for the images
    m1 = mse(imageA, image_1_cent)
    s1 = ssim(imageA, image_1_cent,channel_axis=2)

    m2 = mse(imageA, image_2_cents)
    s2 = ssim(imageA, image_2_cents,channel_axis=2)

    m3 = mse(imageA, image_5_cents)
    s3 = ssim(imageA, image_5_cents,channel_axis=2)
    
    m4 = mse(imageA, image_10_cents)
    s4 = ssim(imageA, image_10_cents,channel_axis=2)
    
    m5 = mse(imageA, image_20_cents)
    s5 = ssim(imageA, image_20_cents,channel_axis=2)
    
    m6 = mse(imageA, image_50_cents)
    s6 = ssim(imageA, image_50_cents,channel_axis=2)
        
    m7 = mse(imageA, image_1_euro)
    s7 = ssim(imageA, image_1_euro,channel_axis=2)
    
    m8 = mse(imageA, image_2_euros)
    s8 = ssim(imageA, image_2_euros,channel_axis=2)
    


    #m1 = mse(imageA, image_1_cent)
    #s1 = ssim(imageA, image_1_cent,multichannel=True)

    #m2 = mse(imageA, image_2_cents)
    #s2 = ssim(imageA, image_2_cents,multichannel=True)

    #m3 = mse(imageA, image_5_cents)
    #s3 = ssim(imageA, image_5_cents,multichannel=True)
    
    #m4 = mse(imageA, image_10_cents)
    #s4 = ssim(imageA, image_10_cents,multichannel=True)
    
   #m5 = mse(imageA, image_20_cents)
    #s5 = ssim(imageA, image_20_cents,multichannel=True)
    
    #m6 = mse(imageA, image_50_cents)
    #s6 = ssim(imageA, image_50_cents,multichannel=True)
    #    
    #m7 = mse(imageA, image_1_euro)
    #s7 = ssim(imageA, image_1_euro,multichannel=True)
    
    #m8 = mse(imageA, image_2_euros)
    #s8 = ssim(imageA, image_2_euros,multichannel=True)
    
    m = max(s1,s2,s3,s4,s5,s6,s7,s8)
    ms2 = min(m1,m2,m3,m4,m5,m6,m7,m8)

    print(m)

    if(m == s1):
        print("1 cent")
        return "Piece1ctm"
    if(m == s2):
        print("2 cents")
        return "Piece2ctms"
    if(m == s3):
        print("5 cents")
        return "Piece5ctms"
    if(m == s4):
        print("10 cents")
        return "Piece10ctms"
    if(m == s5):
        print("20 cents")
        return "Piece20ctms"
    if(m == s6):
        print("50 cents")
        return "Piece50ctms"
    if(m == s7):
        print("1 euro")
        return "Piece1euro"
    if(m == s8):
        print("2 euros")
        return "Piece2euros"

    

    if(ms2 == m1):
        print("1 cent")
        return "Piece1ctm"
    if(ms2 == m2):
        print("2 cents")
        return "Piece2ctms"
    if(ms2 == m3):
        print("5 cents")
        return "Piece5ctms"
    if(ms2 == m4):
        print("10 cents")
        return "Piece10ctms"
    if(ms2 == m5):
        print("20 cents")
        return "Piece20ctms"
    if(ms2 == m6):
        print("50 cents")
        return "Piece50ctms"
    if(ms2 == m7):
        print("1 euro")
        return "Piece1euro"
    if(ms2 == m8):
        print("2 euros")
        return "Piece2euros"
	 

# Charge l'image arg
def check_empty_img(arg):
    img = cv2.imread(str(arg) + ".jpeg")
    if img is None:
        img = cv2.imread(str(arg) + ".jpg")
    if img is None:
        img = cv2.imread(str(arg) + ".png")
    return img

# Combine deux listes
def combine(liste1, liste2):
    combineTmp = np.empty((0,2))
    combineTmp = combineTmp.astype('str')
    first = True
    for i in range(0, len(liste1)):
        tmp = [liste1[i], liste2[i]]
        if(first):
            combineTmp = [tmp]
            first = False
        else:
            combineTmp = np.r_[combineTmp,[tmp]]
    return combineTmp

# Compare deux matrices triées
def compare_array(matrice1, matrice2):
    somme = 0
    for i in range(0, len(matrice1)):
        for j in range(0, len(matrice2)):
            #print('\n \n permier : '+str(matrice1[i][0]))
            #print('\n \n deuxieme : '+str(matrice2[j][0]) +'\n \n')
            if(str(matrice1[i][0]) == str(matrice2[j][0]) and (str(matrice1[i][1]) == str(matrice2[j][1]))):
                somme += 1
    return somme

# Méthode principal
def main():
    # Le nombre de piece bien detecter (le nombre de piece sur l'image correspond bien)
    sommeSuccess = 0
    res = 0
    # Pour chaque image
    for a in range(0, 1,2):
        
        #Récupere l'image
        img = check_empty_img(str(a))
        if img is None :
            continue
        print("Piece : "+str(a))
        kernel = np.ones((5, 5), np.float32) / 25
        imgContour = img.copy() 
        imgBlur = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT) #flou
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY) #passage en gris 
        
        #v = np.median( imgGray )
        #sigma = 0.9
        #lower = int(max(0, (1.0 - sigma) * v))
        #upper = int(min(255, (1.0 + sigma) * v))
        
        threshold1 = 50 #20 50 60
        threshold2 = 60 #80 60 100
        
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2) #Canny pour contour
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1) #Dilatation pour augmenter les contours

        images = getPieces2(imgDil, imgContour, img, a, True)
        #images = getPieces(imgDil, imgContour,img)

        print('\nimages : '+str(len(images)) +'\n')

        #utilisation d'un flou différent (permet d'obtenir d'autre piece) 
        if(len(images) == 0):     
            imgBlur = cv2.GaussianBlur(img, (7, 7), 5)
            imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
            imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

            images = getPieces2(imgDil, imgContour, img, a, False)
            #images = getPieces(imgDil, imgContour,img)

        print('\nimages : '+str(len(images)) +'\n')
        
        show = False
        showResult = True
        saveImgContour = False
        save = False
        saveOther = False
        # Affichage
        if(showResult):
            for i in range(len(images)):
                #imgGray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                #th, im_gray_th_otsu = cv2.threshold(imgGray, 128, 255, cv2.THRESH_OTSU)
                #TEST
                #cv2.imwrite('tmp.jpg', images[i])
                print(type(images[i]))
                #images[i] =test4()
                #compare_images(images[i])
                string = determine_piece(images[i])
                print('string : '+str(string))

                print(len(coin_value_liste))
                
                #if(i>=len(coin_value_liste)):
                    #coin_value_liste = np.vstack([coin_value_liste, string])
                    #tmp = copy.deepcopy(coin_value_liste)
                    #coin_value_liste = np.zeros(len(tmp)+1)
                    #coin_value_liste = coin_value_liste.astype('str')
                    #for ind in range(0, len(tmp)):
                        #coin_value_liste[ind] = tmp[ind]
                    #coin_value_liste[i] = string             
                #else:
                coin_value_liste[i] = string
                
                cv2.imshow(images[i])
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

        if(show):
            cv2.imshow("1 normal",img)
            cv2.imshow("2 Blur",imgBlur)
            cv2.imshow("3 GrayScale",imgGray)
            cv2.imshow("4 Canny",imgCanny)
            cv2.imshow("5 Dilatation",imgDil)
            cv2.imshow("6 Contour",imgContour)

        # Sauvegarde
        if(save):
            #os.mkdir("Result\\"+str(a))
            for i in range(len(images)):
                cv2.imwrite("Result\\"+str(a)+"\piece"+str(i)+".jpg", images[i]) 

        if(saveImgContour) :
             cv2.imwrite(str(a)+".jpg", imgContour)
             
        if(saveOther) :       
            cv2.imwrite(str(a)+"blur.jpg", imgBlur)
            cv2.imwrite(str(a)+"gray.jpg", imgGray)
            cv2.imwrite(str(a)+"canny.jpg", imgCanny)
            cv2.imwrite(str(a)+"dilatation.jpg", imgDil)
            cv2.imwrite(str(a)+"contour.jpg", imgContour)

        #print('\n=======================\n')

        global coin_value_liste_json, coin_liste_json

        combine1 = combine(coin_liste, coin_value_liste)
        combine2 = combine(coin_liste_json, coin_value_liste_json)
        
        print('coin_liste : \n'+str(coin_liste))
        print('coin_value_liste : \n'+str(coin_value_liste))
        print('combine1 : \n'+str(combine1))

        print('\ncoin_liste_json : \n'+str(coin_liste_json))
        print('coin_value_liste_json : \n'+str(coin_value_liste_json))
        print('combine .sjson : \n'+str(combine2))
        
        res += compare_array(combine1, combine2)
        
        print('compteur : '+str(counter_coin))
        print('total coin : ' +str(total_coin))
        print('res : '+str(res))
        
        coin_value_liste_json = np.empty((0,2))
        coin_value_liste_json = coin_value_liste_json.astype('str')
        coin_liste_json = np.empty((0,2))
        coin_liste_json = coin_liste_json.astype('str')

        
        print('\n=======================\n')

        
    print('--------------- ---------------\n')
    print('compteur : '+str(counter_coin))
    print('total coin : ' +str(total_coin))
    print('res : '+str(res))

    print('\n---------------\n')
    print('       taux  ')
    print('\n---------------\n')
    print('         '+str((res/counter_coin)*100)+'%\n')



def compare(original,image_to_compare):
    # 2) Check for similarities between the 2 images
    sift = cv2.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)
            print(len(good_points))
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("GOOD Matches:", len(good_points))
    print("How good it's the match: ", len(good_points) / number_keypoints * 100, "%")

# def test4():
#
#     image = (mpimg.imread('test_images/tmp.jpg').copy() * 255).astype(np.uint8)
#     image_gray = get_image_gray(image)
#     image = check_empty_img("test_images\\tmp")
#     res = resize_image(image_gray,image)
#     #cv2.imshow("piece",res)
#     return res


main()


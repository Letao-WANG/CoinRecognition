import cv2
import numpy as np
import json
import math
import os
from util import *
import matplotlib.pyplot as plt
import copy

import warnings

warnings.simplefilter("error", np.VisibleDeprecationWarning)

#import imagehash
from PIL import Image,ImageChops
from skimage.metrics import structural_similarity as ssim

os.chdir("E:\Repository\CoinRecognition\data") 

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
    return int(((val2-val1)/100) * 100)

# Vérifie si les deux points sont compris dans l'image
def point_in_circle(listeTmp, t3, c):
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
    path = "origin_images\\"+ str(img_num) +".json"
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

    comparing_img = check_empty_img("test_images\\1_ctm")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret1 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret1)

    comparing_img = check_empty_img("test_images\\2_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret2 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret2)

    comparing_img = check_empty_img("test_images\\5_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret3 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret3)

    comparing_img = check_empty_img("test_images\\10_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret4 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret4)

    comparing_img = check_empty_img("test_images\\20_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret5 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret5)

    comparing_img = check_empty_img("test_images\\50_ctms")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret6 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret6)

    comparing_img = check_empty_img("test_images\\1_1")
    comparing_img = cv2.resize(comparing_img, IMG_SIZE)

    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])


    ret7 = cv2.compareHist(target_hist, comparing_hist, 0)
    #print(ret7)

    comparing_img = check_empty_img("test_images\\2_euros")
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
def determine_piece(image_gray):

    IMG_SIZE = (200, 200)

    image = cv2.resize(image_gray, IMG_SIZE)

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
    

    cos_sim1 = calculate_cosine_similarity(image_1_cent, image)

    cos_sim2 = calculate_cosine_similarity(image_2_cents, image)

    cos_sim3 = calculate_cosine_similarity(image_5_cents, image)

    cos_sim4 = calculate_cosine_similarity(image_10_cents, image)

    cos_sim5 = calculate_cosine_similarity(image_20_cents, image)

    cos_sim6 = calculate_cosine_similarity(image_50_cents, image)

    cos_sim7 = calculate_cosine_similarity(image_1_euro, image)

    cos_sim8 = calculate_cosine_similarity(image_2_euros, image)

    print(str(cos_sim1) +" "+ str(cos_sim2) +" "+ str(cos_sim3) +" "+ str(cos_sim4) +" "+ str(cos_sim5) +" "+ str(cos_sim6) +" "+ str(cos_sim7) +" "+ str(cos_sim8))
    m = max(cos_sim1,cos_sim2,cos_sim3,cos_sim4,cos_sim5,cos_sim6,cos_sim7,cos_sim8)
    print(m)

    if(m == cos_sim1):
        print("1 cent")
        return "Piece1ctm"
    if(m == cos_sim2):
        print("2 cents")
        return "Piece2ctms"
    if(m == cos_sim3):
        print("5 cents")
        return "Piece5ctms"
    if(m == cos_sim4):
        print("10 cents")
        return "Piece10ctms"
    if(m == cos_sim5):
        print("20 cents")
        return "Piece20ctms"
    if(m == cos_sim6):
        print("50 cents")
        return "Piece50ctms"
    if(m == cos_sim7):
        print("1 euro")
        return "Piece1euro"
    if(m == cos_sim8):
        print("2 euros")
        return "Piece2euros"
    
# Determine la valeur de la piece en fonction du hash
def determine_piece3(image):

    IMG_SIZE = (200, 200)

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
    

    hash0 = imagehash.average_hash(Image.open("1_ctm_bis"))
    hash1 = imagehash.average_hash(Image.open("test_images\\2_ctms")) 
    cutoff = 5  # maximum bits that could be different between the hashes. 

    if hash0 - hash1 < cutoff:
      print('images are similar')
    else:
      print('images are not similar')



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
    for a in range(0, 60,2):
        
        #Récupere l'image
        img = check_empty_img("origin_images\\"+str(a))
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
                #cv2.imwrite('test_images/tmp.jpg', images[i])
                #images[i] =test4()
                #compare_images(images[i])
                string = determine_piece2(images[i])
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
                
                #cv2.imshow("piece",images[i])
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

def test4():

    image = (mpimg.imread('test_images/tmp.jpg').copy() * 255).astype(np.uint8)
    image_gray = get_image_gray(image)
    image = check_empty_img("test_images\\tmp")
    res = resize_image(image_gray,image)
    #cv2.imshow("piece",res)
    return res


main()

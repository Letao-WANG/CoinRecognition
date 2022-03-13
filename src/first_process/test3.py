import cv2
import numpy as np

ratio = 0.5

# Vérifie si un carré est dans un autre, ne le prend pas en compte
def checkSquareIn(t1):

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
            if x1>x2 and x1<x2f: # x du premier > au xdu second
                if x1f > x2 and x1f<x2f: # x final du premier < au x final du second
                    if y1>y2 and y1<y2f:
                        if y1f>y2 and y1f<y2f:
                            booleen = False
                            

        if booleen :
            t2.append(t1[c])
        else :
            if t1[c] in t2:
                t2.remove(t1[c])
    return t2

# Vérifie si deux carrés sont l'un sur l'autre, si plus de 50%
# est partagé, les réunis, sinon ne fais rien 
def addSquare(t1):
    t2 = []
    t4 = []
    tcopy = []
    for l in range(0,len(t1)) :
        tcopy.append(t1[l])

    doisRefaire = True

    while(doisRefaire):
        doisRefaire = False
        t2 = []
        for c in range(0,len(tcopy)) :
            if doisRefaire:
                break
            booleen = True
            y1 =tcopy[c][1]
            x1 = tcopy[c][0]
            y1f =tcopy[c][3]
            x1f =tcopy[c][2]
            estModifie = False
            lenn = len(tcopy)
            for d in range(0,lenn) :

                x2 = tcopy[d][0]
                y2 =tcopy[d][1]     
                x2f =tcopy[d][2]       
                y2f = tcopy[d][3]
                
                if x2>x1 and x2<x1f:
                    if y2>y1  and y2<y1f:
                        if x2f>x1f:
                            longueur = x1f - x2
                        else:
                            longueur = x2f -x2     

                        if y2f>y1:
                            hauteur = y1f-y2
                        else:
                            hauteur = y2f-y2

                        aire = hauteur * longueur
                        aire1 = (x1f-x1)*(y1f-y1)
                        aire2 = (x2f-x2)*(y2f-y2)
                        ratio1 = aire/aire1
                        ratio2 = aire/aire2
                        if ratio1>=0.5 or ratio2 >= 0.5:
                            newSquare = CombineSquare(tcopy[c],tcopy[d])
                            #t2.append(newSquare)
                            estModifie = True
                            #del t1[d]
                            #del t1[c]
                            #tcopy.remove(tcopy[d])
                            #tcopy.remove(tcopy[c])
                            print(c>=len(tcopy))
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
                            #t4.append(t1[c])
                             #t4.append(t1[d])
                            #t1.pop(d)
                            #t1.pop(c)

        
            if not(estModifie):
                t2.append(tcopy[c])

    return t2

def InTab(t1,element):
    for i in range(0,len(t1)):
        if(element[0] ==  t1[i][0]):
            if (element[1] ==  t1[i][1]):
                if (element[2] ==  t1[i][2]):
                    if (element[3] ==  t1[i][3]):
                        return True
    return False

    

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
    
    
            
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)
    t = []
    ind = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000: #7000
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(str(len(approx)) + " " + str(area))
            x, y, w, h = cv2.boundingRect(approx)

            plusGrand = w
            plusPetit = h
            if h> w:
                plusGrand = h
                plusPetit = w
            if plusPetit/plusGrand >ratio :
                t.append((x,y,x + w, y + h))
            ind+=1

    t3 = checkSquareIn(t)
    t3 = addSquare(t3)
    for c in range(len(t3)):
        cv2.rectangle(imgContour, (t3[c][0],t3[c][1]),(t3[c][2],t3[c][3]) , (0, 255, 0), 5)
        #cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)


# Load image
def check_empty_img(arg):
    img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpeg")
    if img is None:
        img = cv2.imread(r"D:\COur\img_proj\\" + str(arg) + ".jpg")
    if img is None:
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



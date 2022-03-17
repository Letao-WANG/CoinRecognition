import numpy
import math
import cmath
import cv2

from scipy.ndimage import convolve,gaussian_filter
from  matplotlib.pyplot import *

t2 = cv2.imread(r"D:\COur\img_proj\0.jpeg")
thresh = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
count, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
for i in range(1,count):
    t2 = cv2.circle(t2, (int(centroids[i,0]), int(centroids[i,1])), 5, (0, 255, 0, 0), 5)

cv2.imshow('circles', thresh)
cv2.imshow('centers', t2)
cv2.waitKey()

"""
img = imread(r"D:\COur\img_proj\0.jpeg")
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]
array=red*1.0
figure(figsize=(4,4))
imshow(array,cmap=cm.gray)


array = gaussian_filter(array,1)
sobelX = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobelY = numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]])
derivX = convolve(array,sobelX)
derivY = convolve(array,sobelY)
gradient = derivX+derivY*1j
G = numpy.absolute(gradient)
theta = numpy.angle(gradient)

figure(figsize=(8,4))
f,(p1,p2)=subplots(ncols=2)
p1.imshow(derivX,cmap=cm.gray)
p2.imshow(derivY,cmap=cm.gray)

figure(figsize=(4,4))
imshow(G,cmap=cm.gray)

seuil = 0.23
s = G.shape
for i in range(s[0]):
    for j in range(s[1]):
        if G[i][j]<seuil:
            G[i][j] = 0.0
figure(figsize=(4,4))
imshow(G,cmap=cm.gray)

rect = G[1:100,1:100]
figure(figsize=(4,4))
matshow(rect,cmap=cm.gray)

Gmax = G.copy()

def interpolation(array,x,y):
    s = array.shape
    i = math.floor(x)
    j = math.floor(y)
    t = x-i
    u = y-j
    u1 = 1.0-u
    t1 = 1.0-t
    if j==s[0]-1:
        if i==s[1]-1:
            return array[j][i]
        return t*array[j][i]+t1*array[j+1][i]
    if i==s[1]-1:
        return u*array[j][i]+u1*array[j][i+1]
    return t1*u1*array[j][i]+t*u1*array[j][i+1]+\
           t*u*array[j+1][i+1]+t1*u*array[j+1][i]

for i in range(1,s[1]-1):
    for j in range(1,s[0]-1):
        if G[j][i]!=0:
            cos = math.cos(theta[j][i])
            sin = math.sin(theta[j][i])
            g1 = interpolation(G,i+cos,j+sin)
            g2 = interpolation(G,i-cos,j-sin)
            if (G[j][i]<g1) or (G[j][i]<g2):
                Gmax[j][i] = 0.0
                
figure(figsize=(6,6))
imshow(Gmax,cmap=cm.gray)

Gmax_2 = G.copy()
pi = math.pi
a = numpy.zeros(4)
a[0] = pi/8
for k in range(1,4):
    a[k] = a[k-1]+pi/4
for j in range(1,s[0]-1):
    for i in range(1,s[1]-1):
        if G[j][i]!=0:
            b = theta[j][i]
            if b>0:
                if (b<a[0]) or (b>a[3]):
                    g1 = G[j][i+1]
                    g2 = G[j][i+1]
                elif (b<a[1]):
                    g1 = G[j+1][i+1]
                    g2 = G[j-1][i-1]
                elif (b<a[2]):
                    g1 = G[j+1][i]
                    g2 = G[j-1][i]
                else:
                    g1 = G[j+1][i-1]
                    g2 = G[j-1][i+1]
            elif b<0:
                if (b<-a[3]):
                    g1 = G[j][i+1]
                    g2 = G[j][i-1]
                elif (b<-a[2]):
                    g1 = G[j-1][i-1]
                    g2 = G[j+1][i+1]
                elif (b<-a[1]):
                    g1 = G[j-1][i]
                    g2 = G[j+1][i]
                elif (b<-a[0]):
                    g1 = G[j-1][i+1]
                    g2 = G[j+1][i-1]
                else:
                    g1 = G[j][i+1]
                    g2 = G[j][i-1]
            if (G[j][i]<g1) or (G[j][i]<g2):
                Gmax_2[j][i] = 0.0
            
                
figure(figsize=(6,6))
imshow(Gmax_2,cmap=cm.gray)         



Gfinal = Gmax.copy()
Gfinal_2 = Gmax_2.copy()
seuil = 0.2
for j in range(s[0]):
    for i in range(s[1]):
        if Gfinal[j][i]<seuil:
            Gfinal[j][i] = 0.0
        else:
            Gfinal[j][i] = 255.0
        if Gfinal_2[j][i]<seuil:
            Gfinal_2[j][i] = 0.0
        else:
            Gfinal_2[j][i] = 255.0
figure(figsize=(10,6))
f,(p1,p2)=subplots(ncols=2)
p1.imshow(Gfinal,cmap=cm.gray)
p2.imshow(Gfinal_2,cmap=cm.gray)
show()
"""

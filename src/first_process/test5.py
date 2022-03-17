import cv2
import numpy as np

image =cv2.imread(r"D:\COur\img_proj\44.png")
imgBlur = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_DEFAULT)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100
params.maxArea = 100000000000000000000

params.filterByCircularity = True
params.minCircularity = 0.005

params.filterByConvexity = True
params.minConvexity = 0.3

params.filterByInertia = True
params.minInertiaRatio = 0.4

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(imgGray, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("Original Image",image)
cv2.imshow("Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
image_rgb = plt.imread(r"D:\COur\img_proj\0.jpeg")
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=2.0,
              low_threshold=0.1, high_threshold=0.2)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                sharey=True,
                                subplot_kw={'adjustable':'box'})

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()


import cv2 #opencv
import matplotlib.pyplot as plt
import matplotlib.cm as cm


img = plt.imread(r"D:\COur\img_proj\0.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


plt.subplot(2,3,6)
plt.axis('off') 
plt.title("Canny", fontsize=10)
dst = cv2.Canny(gray,threshold1 = 100,threshold2 = 500)
#Les plus petits seuils 1 et 2 sont utilisés pour joindre les bords.
#Les plus grands sont utilisés pour la détection initiale des bords plus forts.

plt.imshow(dst,cmap=cm.gray)
plt.show()"""

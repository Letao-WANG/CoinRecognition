import cv2
import numpy as np
 
# Load image
def check_empty_img(arg):
    img = cv2.imread(r"D:\COur\img_proj\\"+str(arg)+".jpeg")
    if img is None:
        img = cv2.imread(r"D:\COur\img_proj\\"+str(arg)+".jpg")
    elif img is None:
        img = cv2.imread(r"D:\COur\img_proj\\"+str(arg)+".png")
    return img

#On utilisera aprés les fichiers JSON 
docsTrueValue = (3,2,1,1,1,7,1,2,2,1,1,5,1,5,5,1,1,1,2,3,1,1,1,1,1,1,1,1,1,1)

#best b = 4
for b in range(5,10) :
    truevaleur = 0
    for a in range(0,30) :

        image = check_empty_img(a)
        if image is None :
            continue
     
        # Set our filtering parameters
        # Initialize parameter settiing using cv2.SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
         
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 450
        params.maxArea = 1000000
         
        # Set Circularity filtering parameters
        params.filterByCircularity = True
        params.minCircularity = 0
        params.maxCircularity = 1
         
        # Set Convexity filtering parameters
        params.filterByConvexity = True
        params.minConvexity = b*0.1
        params.maxConvexity = 1
             
        # Set inertia filtering parameters
        params.filterByInertia = True
        params.minInertiaRatio = b*0.1
        params.maxInertiaRatio = 1
         
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
             
        # Detect blobs
        keypoints = detector.detect(image)
         
        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
        number_of_blobs = len(keypoints)
        #text = "Number of Circular Blobs: " + str(len(keypoints))
        #cv2.putText(blobs, text, (20, 550),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        #if number_of_blobs != 0:
        #    isFind[a] = 1
        r = (docsTrueValue[a] == number_of_blobs)
        if r :
            truevaleur += 1
            print("image numéro " + str(a) + " : "+ str(number_of_blobs) + " "+ str(r))
        else :
            print("image numéro "  +str(a) + " : no")
        # Show blobs
        #cv2.imshow("Filtering Circular Blobs Only", blobs)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    print(truevaleur)

from cv2 import SURF, imread

import cv2
import scipy.misc

img = imread('train-10/Suburb/image_0029.jpg',0)
surf = SURF(400)
kp, des = surf.detectAndCompute(img, None)

im = cv2.imread('train-10/Suburb/image_0029.jpg')
cv2.imshow('original',im)

#s = cv2.SIFT() # SIFT
s = cv2.SURF() # SURF
keypoints = s.detect(im)

for k in keypoints:
    cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)
    #cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)


cv2.imshow('SURF_features',im)
cv2.waitKey()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:19:33 2017

@author: tushar
"""
import cv2
import numpy as np
from sklearn import svm

import feature_extractor

#clf = svm.SVC(gamma=0.001, C=100.)

img = cv2.imread("tennis_ball.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.medianBlur(img, 45)
blur_gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

#height, width = img.shape[:2]

d_img = np.reshape(gray_img, (1,np.product(gray_img.shape)))

height, width = d_img.shape[:2]

# get features
# features structure:
# - @0: yellowness
# - @1: circle existence
# - @2: seam probability
features = []

#features.append(feature_extractor.extractYellowness(img))
#features.append(feature_extractor.extractBestEstimatedCircle(blur_gray_img))
features.append(feature_extractor.extractSeam(img))

features.clear

#DEBUG    #---------------------------------------------------------------------------
#print(str(height) + "x" + str(width))

#cv2.imshow('blurred gray image', blur_gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    #---------------------------------------------------------------------------

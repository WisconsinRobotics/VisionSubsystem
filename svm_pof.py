# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:19:33 2017

@author: tushar
"""
import cv2
import numpy as np
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

img = cv2.imread("tennis_ball.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#height, width = img.shape[:2]

d_img = np.reshape(gray_img, (1,np.product(gray_img.shape)))

height, width = d_img.shape[:2]

print(str(height) + "x" + str(width)) 


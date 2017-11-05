# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:43:35 2017

@author: tushar
"""

import cv2
import numpy as np

def extractYellowness(image):
    norm_img = image
    #cv2.normalize(image, norm_img, alpha=0, beta=20, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #hist = cv2.calcHist([norm_img], [0], None, [2], [0,20])
    #back_project = cv2.calcBackProject([norm_img], [0], hist, [0,20], 1)

    #filtered_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_RAINBOW)

    #DEBUG: display image
    cv2.imshow('filtered image', norm_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

def extractCircles(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=350, maxRadius=500)
    circles = np.uint16(np.around(circles))

    #DEBUG: draw circles
    #for i in circles[0,:]:
    #    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    #cv2.imshow('detected circles', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return circles

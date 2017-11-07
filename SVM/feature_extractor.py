# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:43:35 2017

@author: tushar
"""

import cv2
import numpy as np

def extractYellowness(image):
    """ Get some measureness of the "yellowness" of an image.
    :param image: Input image to get "yellowness" of, should be in grayscale.
    """

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([33,50,50])
    upper_yellow = np.array([37,255,255])
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    test_img = hsv_img.copy()
    test_img[np.where(mask==0)] = 0

    analysis_img = cv2.cvtColor(test_img, cv2.COLOR_HSV2BGR)
    gray_analysis_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    num_yellow = cv2.countNonZero(gray_analysis_img)
    height, width, channels = image.shape
    total_size = height * width
    yellowness = num_yellow / total_size

    #DEBUG: display image
    #cv2.imshow('original image', image)
    #print('size: ', total_size, ', yellow: ', num_yellow, ', yellowness: ', yellowness)
    #cv2.imshow('hsv image', hsv_img)
    #cv2.imshow('new image', test_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return yellowness

def extractCircles(image):
    """ Get circles detected in an image. Uses Hough Transform.
    :param image: Input image to get circles from, should be in grayscale and blurred.
    """
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=350, maxRadius=500)
    circles = np.uint16(np.around(circles))

    #DEBUG: draw circles
    #for i in circles[0,:]:
    #    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    #cv2.imshow('detected circles', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if len(circles) > 0:
        return 1
    else:
        return 0

def main(image_set, labels):
    file = open("data.txt", 'w')
    features = []
    for i in range(len(image_set)):
        features.append(extractCircles(image_set[i]))
        features.append(extractYellowness(image_set[i]))
        file.write(labels[i] + "," + features[0] + "," + features[1] + "\n")        
        features.clear
        

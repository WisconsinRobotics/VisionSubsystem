# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:43:35 2017

@author: tushar
"""

import sys
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import deque

# Helper functions here:
#-------------------------------------------------------------------------------
def getEdgeCoords(edge_img, x_start, y_start, r_avg, thresh):
    """
    Step types follow compass directions:
      - 0 = N
      - 1 = NE
      - 2 = E
      - 3 = SE
      - 4 = S
      - 5 = SW
      - 6 = W
      - 7 = NW

    :param edge_img: Input image, must be edge image
    :param r_avg: Radius of the averaged circle, for determining valid pts
    :param step_type: Which direction this step is, see above
    :param is_start: Determine if we need to do first steps or not
    """

    print("at beginning of getEdgeCoords")

    d = deque()
    edge_pts = []
    h = len(edge_img)
    w = len(edge_img[0])

    d.appendleft([x_start, y_start + 1, 0, 1])
    d.appendleft([x_start + 1, y_start + 1, 1, 1])
    d.appendleft([x_start + 1, y_start, 2, 1])
    d.appendleft([x_start + 1, y_start - 1, 3, 1])
    d.appendleft([x_start, y_start - 1, 4, 1])
    d.appendleft([x_start - 1, y_start - 1, 5, 1])
    d.appendleft([x_start - 1, y_start, 6, 1])
    d.appendleft([x_start - 1, y_start + 1, 7, 1])
    print(d)

    while (d):
        info = d.pop()
        if (edge_img[info[0], info[1]] == 1):
            print("found")
            sys.exit()
        if ((info[0] < 0) or (info[0] > (w - 1)) or (info[1] < 0) or (info[1] > (h - 1))):
            continue
        elif (edge_img[info[0], info[1]] == 1):
            print("found point")
            if ((info[3] > (r_avg - thresh)) or (info[3] < (r_avg + thresh))):
                print("adding point")
                edge_pts.append([info[0], info[1]])
            else:
                if (info[2] == 0):
                    print("at 0 case")
                    d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])
                elif (info[2] == 1):
                    print("at 1 case")
                    d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])
                    d.appendleft([info[0] + 1, info[1] + 1, 1, info[3] + 1])
                    d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
                elif (info[2] == 2):
                    print("at 2 case")
                    d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
                elif (info[2] == 3):
                    print("at 3 case")
                    d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
                    d.appendleft([info[0] + 1, info[1] - 1, 3, info[3] + 1])
                    d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
                elif (info[2] == 4):
                    print("at 4 case")
                    d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
                elif (info[2] == 5):
                    print("at 5 case")
                    d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
                    d.appendleft([info[0] - 1, info[1] - 1, 5, info[3] + 1])
                    d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
                elif (info[2] == 6):
                    print("at 6 case")
                    d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
                else:
                    print("at 7 case")
                    d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
                    d.appendleft([info[0] - 1, info[1] + 1, 7, info[3] + 1])
                    d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])
        else:
            if (info[2] == 0):
                d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])
            elif (info[2] == 1):
                d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])
                d.appendleft([info[0] + 1, info[1] + 1, 1, info[3] + 1])
                d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
            elif (info[2] == 2):
                d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
            elif (info[2] == 3):
                d.appendleft([info[0] + 1, info[1], 2, info[3] + 1])
                d.appendleft([info[0] + 1, info[1] - 1, 3, info[3] + 1])
                d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
            elif (info[2] == 4):
                d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
            elif (info[2] == 5):
                d.appendleft([info[0], info[1] - 1, 4, info[3] + 1])
                d.appendleft([info[0] - 1, info[1] - 1, 5, info[3] + 1])
                d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
            elif (info[2] == 6):
                d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
            else:
                d.appendleft([info[0] - 1, info[1], 6, info[3] + 1])
                d.appendleft([info[0] - 1, info[1] + 1, 7, info[3] + 1])
                d.appendleft([info[0], info[1] + 1, 0, info[3] + 1])

    return edge_pts

# Feature extractors here:
#-------------------------------------------------------------------------------

def extractYellowness(image):
    """
    Get some measureness of the "yellowness" of an image.

    Concepts and Ideas:
    - better constraints on desired color
    - search for best range for detecting color

    :param image: Input image to get "yellowness" of, should be in grayscale.
    """

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # from very shallow experimentation, I have found that the Hue range of 33 - 37 gives a nice range of tennis-ball-y colors
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
    #---------------------------------------------------------------------------
    #cv2.imshow('original image', image)
    #print('size: ', total_size, ', yellow: ', num_yellow, ', yellowness: ', yellowness)
    #cv2.imshow('hsv image', hsv_img)
    #cv2.imshow('new image', test_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    #---------------------------------------------------------------------------

    return yellowness

def extractCircles(image):
    """
    Get "goodness" of circles detected in an image. Uses Hough Transform.

    We define the "goodness" of a circle object in an image to be a measure of how close an averaged circle matches its corresponding object. This averaged circle is simply the circle calculated from the average of all detected centers and radii. A select number of points (we have chosen 1000) are taken on the circle and the distance between those points and nearest edge point of the object (radially) is calculated and normalized. This normalized value is our "goodness" value and is returned.

    Some examples of good and bad "goodness" values include:

    TODO: give examples here

    :param image: Input image to get circles from, should be in grayscale and blurred.
    """
    # get edge image of input
    edges = cv2.Canny(image, 1, 25)

    #DEBUG: copy images and show edge image    #---------------------------------------------------------------------------
    test_img = image.copy()
    test_edge_img = edges.copy()
    edges_debug = edges.copy()
    #plt.subplot(121), plt.imshow(image, cmap = "gray")
    #plt.title("Original Image")#, plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(edges, cmap = "gray")
    #plt.title("Edge Image")#, plt.xticks([]), plt.yticks([])
    #plt.show()    #---------------------------------------------------------------------------

    # get circles using Hough Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=360, maxRadius=450)
    if len(circles) == 0:
        print("No circles detected")
        return 0
    circles = np.uint16(np.around(circles))

    #DEBUG: draw circles    #---------------------------------------------------------------------------
    #for i in circles[0,:]:
    #    cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 0), 2)
    #    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 0), 3)
    #    cv2.circle(edges, (i[0], i[1]), i[2], (255, 255, 255), 2)
    #    cv2.circle(edges, (i[0], i[1]), 2, (255, 255, 255), 3)
    #cv2.imshow("detected circles", image)
    #cv2.waitKey(0)
    #cv2.imshow("detected circles with edge image", edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    #---------------------------------------------------------------------------

    # average all detected circles w/in a range
    x_tot = 0
    y_tot = 0
    r_tot = 0
    for x in circles[0,:]:
        x_tot += x[0]
        y_tot += x[1]
        r_tot += x[2]
    x_avg = int(round(x_tot/(len(circles) + 1)))
    y_avg = int(round(y_tot/(len(circles) + 1)))
    r_avg = int(round(r_tot/(len(circles) + 1)))

    #DEBUG: draw averaged circles
    #---------------------------------------------------------------------------
    #print("x_tot: ", x_tot, " y_tot: ", y_tot, " r_tot: ", r_tot)
    print("x_avg: ", x_avg, " y_avg: ", y_avg, " r_avg: ", r_avg)
    #cv2.circle(test_img, (x_avg, y_avg), r_avg, (0, 0, 0), 2)
    #cv2.circle(test_img, (x_avg, y_avg), 2, (0, 0, 0), 3)
    cv2.circle(test_edge_img, (x_avg, y_avg), r_avg, (255, 255, 255), 2)
    cv2.circle(test_edge_img, (x_avg, y_avg), 2, (255, 255, 255), 3)
    #cv2.imshow("averaged circle", test_img)
    #cv2.waitKey(0)
    cv2.imshow("averaged circle with edge image", test_edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    # calculate "goodness"
    normalization_factor = 1000
    offsets = []

    search_range = 15
    print("now calling getEdgeCoords")
    edge_coords = getEdgeCoords(edges_debug, x_avg, y_avg, r_avg, search_range)
    print("finished calling getEdgeCoords")

    #DEBUG: show detected coordinates
    #---------------------------------------------------------------------------
    for i in edge_coords:
        print("detected edge x-coordinate: ", edge_coord[i][0], " y-coordinate: ", edge_coord[i][1])
    #---------------------------------------------------------------------------

    offsets_tot = 0
    max_offset = max(offsets)
    for x in offsets[0,:]:
        offsets_tot += math.fabs(x - max_offset)
    offsets_avg = offsets_tot / (len(offsets) + 1)
    goodness = offsets_avg / normalization_factor

    #DEBUG: check goodness and relevant values
    #---------------------------------------------------------------------------
    print("max offset: ", max_offset, " offsets_tot: ", offsets_tot, " offsets_avg: ", offsets_avg)
    print("goodness: ", goodness)
    #---------------------------------------------------------------------------

    return goodness

def extractSeam():
    """
    Get probability of seams on detected object. Must use some tennis ball detection function first.

    Using gradient orientation on the remaining image to find seams seems like a potential way forward.
    """


def extractContrast():
    """
    Get best fitting radial gradient difference orientations. For example, a tennis ball would have many vectors pointing away from its edge due to the contrast between its lighter colors and the surrounding darker colors.

    See Figure 2 of http://epubs.surrey.ac.uk/733265/1/BMVC05.pdf.
    """


def extractFuzzyness():
    """
    Get "goodness" of fuzzyness on detected objects. Must use some tennis ball detection, or other object-isolating, function first.

    Using edge detection and then selecting the highest amount of edge pixels in each object's region of the image could be a possible way forward.
    """


def main(image_set, labels):
    file = open("data.txt", 'w')
    features = []
    for i in range(len(image_set)):
        features.append(extractCircles(image_set[i]))
        features.append(extractYellowness(image_set[i]))
        file.write(labels[i] + "," + features[0] + "," + features[1] + "\n")
        features.clear

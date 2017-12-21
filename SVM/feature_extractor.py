# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:43:35 2017

@author: tushar
"""

import sys
import cv2
import numpy as np
import math
import statistics
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
    d = deque()
    edge_pts = []
    h = len(edge_img)
    w = len(edge_img[0])

    d.appendleft([x_start, y_start + 1, 0])
    d.appendleft([x_start + 1, y_start + 1, 1])
    d.appendleft([x_start + 1, y_start, 2])
    d.appendleft([x_start + 1, y_start - 1, 3])
    d.appendleft([x_start, y_start - 1, 4])
    d.appendleft([x_start - 1, y_start - 1, 5])
    d.appendleft([x_start - 1, y_start, 6])
    d.appendleft([x_start - 1, y_start + 1, 7])

    while (d):
        info = d.pop()
        if ((info[0] < 0) or (info[0] > (h - 1)) or (info[1] < 0) or (info[1] > (w - 1))):
            continue
        elif (edge_img[info[0], info[1]] == 255):
            length = math.hypot(x_start - info[0], y_start - info[1])
            if ((length > (r_avg - thresh)) and (length < (r_avg + thresh))):
                edge_pts.append([info[0], info[1]])
            else:
                if (info[2] == 0):
                    d.appendleft([info[0], info[1] + 1, 0])
                elif (info[2] == 1):
                    d.appendleft([info[0], info[1] + 1, 0])
                    d.appendleft([info[0] + 1, info[1] + 1, 1])
                    d.appendleft([info[0] + 1, info[1], 2])
                elif (info[2] == 2):
                    d.appendleft([info[0] + 1, info[1], 2])
                elif (info[2] == 3):
                    d.appendleft([info[0] + 1, info[1], 2])
                    d.appendleft([info[0] + 1, info[1] - 1, 3])
                    d.appendleft([info[0], info[1] - 1, 4])
                elif (info[2] == 4):
                    d.appendleft([info[0], info[1] - 1, 4])
                elif (info[2] == 5):
                    d.appendleft([info[0], info[1] - 1, 4])
                    d.appendleft([info[0] - 1, info[1] - 1, 5])
                    d.appendleft([info[0] - 1, info[1], 6])
                elif (info[2] == 6):
                    d.appendleft([info[0] - 1, info[1], 6])
                else:
                    d.appendleft([info[0] - 1, info[1], 6])
                    d.appendleft([info[0] - 1, info[1] + 1, 7])
                    d.appendleft([info[0], info[1] + 1, 0])
        else:
            if (info[2] == 0):
                d.appendleft([info[0], info[1] + 1, 0])
            elif (info[2] == 1):
                d.appendleft([info[0], info[1] + 1, 0])
                d.appendleft([info[0] + 1, info[1] + 1, 1])
                d.appendleft([info[0] + 1, info[1], 2])
            elif (info[2] == 2):
                d.appendleft([info[0] + 1, info[1], 2])
            elif (info[2] == 3):
                d.appendleft([info[0] + 1, info[1], 2])
                d.appendleft([info[0] + 1, info[1] - 1, 3])
                d.appendleft([info[0], info[1] - 1, 4])
            elif (info[2] == 4):
                d.appendleft([info[0], info[1] - 1, 4])
            elif (info[2] == 5):
                d.appendleft([info[0], info[1] - 1, 4])
                d.appendleft([info[0] - 1, info[1] - 1, 5])
                d.appendleft([info[0] - 1, info[1], 6])
            elif (info[2] == 6):
                d.appendleft([info[0] - 1, info[1], 6])
            else:
                d.appendleft([info[0] - 1, info[1], 6])
                d.appendleft([info[0] - 1, info[1] + 1, 7])
                d.appendleft([info[0], info[1] + 1, 0])

    return edge_pts

# Feature extractors here:
#-------------------------------------------------------------------------------

def extractYellowness(image):
    """
    Get some measurement of the "yellowness" of an image.

    NOTE: a potential big improvement would be to find regions where there is a high concentration of 
    filtered yellow pixels before using some bounding area around that region as the analyzed "image"

    :param image: Input image to get "yellowness" of, should be in grayscale.
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # setup mask for image
    # NOTE: from very shallow experimentation, I have found that the Hue range of 33 - 37 gives 
    # a nice range of tennis ball-y colors
    lower_yellow = np.array([33,175,0])
    upper_yellow = np.array([37,255,255])
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    
    hsv_img[np.where(mask==0)] = 0

    analysis_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    gray_analysis_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)
    num_yellow = cv2.countNonZero(gray_analysis_img)
    height, width, channels = image.shape
    total_size = height * width
    yellowness = num_yellow / total_size

    #DEBUG: display image
    #---------------------------------------------------------------------------
    print('size: ', total_size, ', yellow: ', num_yellow, ', yellowness: ', yellowness)
    cv2.imshow('analysis image', hsv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    return yellowness

def extractBestEstimatedCircle(image):
    """
    Get "goodness" of circles detected in an image. Uses Hough Transform.

    We define the "goodness" of a circle object in an image to be a measure of how close an averaged circle 
    matches its corresponding object. This averaged circle is simply the circle calculated from the 
    average of all detected centers and radii. A select number of points (we have chosen 1000) are taken 
    on the circle and the distance between those points and nearest edge point of the object (radially) 
    is calculated and normalized. This normalized value is our "goodness" value and is returned.

    NOTE: a potential big improvement would be to find regions where there is a high concentration of 
    circle centers and select the highest concentration region, or simply to limit the total number of 
    accepted circles and run the circularity rating on each one of them

    :param image: Input image to get circles from, should be in grayscale and blurred.
    """
    # get edge image of input
    edge_img = cv2.Canny(image, 1, 25)
    
    cv2.imshow('edge image', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get circles using Hough Transform
    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, 1, 5, param1=50, param2=30, minRadius=10, maxRadius=40)
    if len(circles) == 0:
        print("No circles detected")
        return 0
    circles = np.uint16(np.around(circles))

    #DEBUG: draw circles
    #---------------------------------------------------------------------------
    test_edge_img = edge_img.copy()
    for i in circles[0,:]:
        print("x: ", i[0], " y: ", i[1], " r: ", i[2])
        cv2.circle(test_edge_img, (i[0], i[1]), i[2], (255, 255, 255), 2)
        cv2.circle(test_edge_img, (i[0], i[1]), 2, (255, 255, 255), 3)
    cv2.imshow("detected circles with edge image", test_edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    # average all detected circles w/in a range
    x_tot = 0
    y_tot = 0
    r_tot = 0
    for x in circles[0,:]:
        x_tot += x[0]
        y_tot += x[1]
        r_tot += x[2]
    x_avg = int(round(x_tot/(len(circles[0,:]))))
    y_avg = int(round(y_tot/(len(circles[0,:]))))
    r_avg = int(round(r_tot/(len(circles[0,:]))))

    #DEBUG: draw averaged circles
    #---------------------------------------------------------------------------
    test_edge_img = edge_img.copy()
    print("x_avg: ", x_avg, " y_avg: ", y_avg, " r_avg: ", r_avg)
    cv2.circle(test_edge_img, (x_avg, y_avg), r_avg, (255, 255, 255), 2)
    cv2.circle(test_edge_img, (x_avg, y_avg), 2, (255, 255, 255), 3)
    cv2.imshow("averaged circle with edge image", test_edge_img)
    cv2.waitKey(0)
    #---------------------------------------------------------------------------

    # calculate "goodness"
    normalization_factor = 100
    offsets = []
    search_range = math.ceil(r_avg / 10)

    edge_coords = getEdgeCoords(edge_img, x_avg, y_avg, r_avg, search_range)

    #DEBUG: show detected coordinates
    #---------------------------------------------------------------------------
    for i in edge_coords:
        print("detected edge x-coordinate: ", i[0], " y-coordinate: ", i[1])
        cv2.circle(test_edge_img, (i[0], i[1]), 20, (255, 255, 255), 1)
        cv2.imshow("averaged circle with edge image", test_edge_img)
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    for x in edge_coords:
        offset = math.fabs(r_avg - math.hypot(x_avg - x[0], y_avg - x[1]))
        offsets.append(offset)

    if len(offsets) == 0:
        print("No offsets successfully processed")
        return 0
        
    offsets_tot = 0
    max_offset = max(offsets)
    for x in offsets:
        offsets_tot += math.fabs(x - max_offset)
    offsets_avg = offsets_tot / (len(offsets) + 1)
    goodness = offsets_avg / normalization_factor

    #DEBUG: check goodness and relevant values
    #---------------------------------------------------------------------------
    print("max offset: ", max_offset, " offsets_tot: ", offsets_tot, " offsets_avg: ", offsets_avg)
    print("goodness: ", goodness)
    #---------------------------------------------------------------------------

    return goodness

def extractSeam(image):
    """
    Get probability of seams on detected object. One way of doing this is to use the HSV version of the image. 
    Then, it may be possible to select only a certain range of pixel values that correspond to tennis ball seams 
    (this can be relative to the rest of the picture). After getting this range, the picture is converted such 
    that only that range is displayed, and a final calculation is made grading the amount of those remaining 
    pixels. A necessary extension requires that the remainder image be converted into an edge image and similar 
    curves detected (as the edges of a seam would appear).
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    equ_v = cv2.equalizeHist(v)
    equ_hsv_img = cv2.merge((h, s, equ_v))

    # setup mask for image
    #lower_seam_range = np.array([0,10,0])
    #upper_seam_range = np.array([60,160,150])
    lower_seam_range = np.array([33,175,0])
    upper_seam_range = np.array([37,255,255])
    mask = cv2.inRange(equ_hsv_img, lower_seam_range, upper_seam_range)

    equ_hsv_img[np.where(mask==0)] = 0
    
    cv2.imshow("hsv img", hsv_img)
    cv2.waitKey(0)
    cv2.imshow("equ img", equ_hsv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    equ_h_val, equ_s_val, equ_v_val = cv2.split(equ_hsv_img)
    ret, bin_img = cv2.threshold(equ_s_val, .001, 255, cv2.THRESH_BINARY)
    cnt_img, contours, hrch = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # the contour length threshold is a relatively arbitrary value based on what behavior I've noticed in test image contours
    max_cnt_len = len(max(contours, key=len))
    min_cnt_len = len(min(contours, key=len))
    cnt_thresh = (min_cnt_len + (.05 * max_cnt_len)) / 2

    rows, cols = bin_img.shape[:2]
    line_params = []
    for x in contours:
        epsilon = .005 * cv2.arcLength(x, True)
        approx = cv2.approxPolyDP(x, epsilon, True)
        if len(x) > cnt_thresh:
            [vx, vy, x_int, y_int] = cv2.fitLine(x, cv2.DIST_L2, 0, .01, .01)
            #DEBUG: draw approximated contours and best fit lines
            #---------------------------------------------------------------------------
            print("epsilon: ", epsilon)
            print("approx info: ", approx, " length: ", len(approx))
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
            cv2.imshow("contours on image", image)
            cv2.waitKey(0)
            
            left_val = int((-x_int * vy / vx) + y_int)
            right_val = int(((cols - x_int) * vy / vx) + y_int)
            cv2.line(image, (cols - 1, right_val), (0, left_val), (0, 255, 0), 2)
            cv2.imshow("contours on image", image)
            cv2.waitKey(0)
            #---------------------------------------------------------------------------
            line_params.append([vx, vy])
    cv2.destroyAllWindows()

    angles = []
    ang_range = math.pi / 12
    for x in line_params:
        m = x[1] / x[0]
        angle = math.atan(m)
        angles.append(angle)
    median_ang = statistics.median(angles)

    seam_prob = 0
    bfl_val = 0
    for y in angles:
        if ((y > (median_ang - ang_range)) and (y < (median_ang + ang_range))):
            bfl_val += 1

    # again, a relatively arbitrary estimate of what this probability would look like given these 
    # contributing values, the normalization factor can (and probably should) be adjusted; note that 
    # bfl_val has significantly more weight than matched_shape_val
    normalization_factor = 50
    seam_prob = bfl_val / normalization_factor

    return seam_prob

def extractGreatestCircularContrast():
    """
    Get best fitting radial gradient difference orientations. For example, a tennis ball would have many vectors pointing away from its edge due to the contrast between its lighter colors and the surrounding darker colors.

    See Figure 2 of http://epubs.surrey.ac.uk/733265/1/BMVC05.pdf.
    """


def extractBestFuzzyObject():
    """
    Get "goodness" of fuzzyness on detected objects. Must use some tennis ball detection, or other object-isolating, function first.

    Using edge detection and then selecting the highest amount of edge pixels in each object's region of the image could be a possible way forward.
    """


def main(image_set, labels):
    file = open("data.txt", 'w')
    features = []
    for i in range(len(image_set)):
        features.append(extractBestEstimatedCircle(image_set[i]))
        features.append(extractYellowness(image_set[i]))
        file.write(labels[i] + "," + features[0] + "," + features[1] + "\n")
        features.clear

import tensorflow as tf
import math
import cv2
import numpy as np
import time
import os
from collections import deque

# "constants"
FRAME_RATE = 15
MAX_STACK_SIZE = 60
FT_TO_M_CONV_FACTOR = .3048        # [m/ft] obviously
M_TO_PX_CONV_FACTOR = 3779.5276    # [px/m] obviously
CAM_FOCAL_L = 112.0                # what unit is this?
TB_KNOWN_W = 2.6                   # [in]
IMG_W = 1920
IMG_H = 1080

frame_stack = []
output_info = []

# cv2 information
# "constants"
CIRCLE_MIN_R = 2
KERNEL_SIZE = 15

min_HSV = np.array([25,50,0])
max_HSV = np.array([75,255,255])

# return information
x_center_final = None
y_center_final = None
r_final = None
tb_dist_final = None
detect_timestamp = None

temp_path = r".\tb_test_images"
debug = True
image_debug = False
advanced_debug = False

# start Tensorflow cnn session
sess = tf.Session()

def getNetDetections(image):
    # preprocess images
    predict_data = np.zeros(shape=(1, 150, 150, 3))
    resized_img = cv2.resize(image, (150, 150))
    resized_img = (resized_img / (np.max(resized_img)/2)) - 1
    predict_data[0] = resized_img
    predict_data = predict_data.astype(np.float32)

    assert not np.any(np.isnan(predict_data))

    predict_data = predict_data.flatten()

    # DEBUG
    if debug:
        print("original shape: ")
        print(type(predict_data))
        print(predict_data.shape)
        print("flattened shape: ")
        print(type(predict_data))
        print(predict_data.shape)
        #exit()

    # predict
    full_model_dir = "./tb_cnn_model_serve/1524550001"
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
    predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
    model_input = tf.train.Example(features=tf.train.Features(feature={"x": tf.train.Feature(float_list=tf.train.FloatList(value=predict_data))}))
    model_input = model_input.SerializeToString()
    output_dict = predictor({"predictor_inputs":[model_input]})
    if 1 == output_dict["classes"][0]:
        sureness = output_dict["probabilities"][0][1]
        if debug: print("predicted - contains tennis ball with sureness: ", sureness)
        if sureness >= 0.6:
            return True
    else:
        if debug: print("predicted - does NOT contains tennis ball")
        return False

def getHsvMask(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    temp_img = hsv_img.copy()

    # setup mask for image
    mask = cv2.inRange(hsv_img, min_HSV, max_HSV)

    # erode and dilate mask
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    hsv_img[np.where(mask==0)] = 0

    # DEBUG: display image
    #---------------------------------------------------------------------------
    if image_debug:
        print(" *** DEBUG: display hsv/masked images")
        cv2.imshow('image', image)
        cv2.imshow('hsv image', temp_img)
        cv2.imshow('masked image', hsv_img)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    return temp_img, hsv_img


def getBallLoc(image):
    # get circles using Hough Transform
    # NOTE: - param1 = higher threshold to Canny (apparently)
    #       - param2 = accumulator threshold, how large the pool is for similar circle centers
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 5, param1=255, param2=20, minRadius=5, maxRadius=0)
    if circles is None:
        print("No circles detected")

        return 0, 0, 0, 0
    circles = np.uint16(np.around(circles))

    # DEBUG: draw circles
    #---------------------------------------------------------------------------
    if advanced_debug:
        print(" *** DEBUG: draw circles")
        test_img = image.copy()
        for i in circles[0,:]:
            print("x: ", i[0], " y: ", i[1], " r: ", i[2])
            cv2.circle(test_img, (i[0], i[1]), i[2], (255, 255, 255), 2)
            cv2.circle(test_img, (i[0], i[1]), 2, (255, 255, 255), 3)
        print(len(circles[0,:]))
        cv2.imshow("detected circles with edge image", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #exit()
    #---------------------------------------------------------------------------

    # get best circle
    x_best = circles[0,0,0]
    y_best = circles[0,0,1]
    r_best = circles[0,0,2]

    # -OR-

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

    # DEBUG: draw averaged circles
    #---------------------------------------------------------------------------
    if image_debug:
        print(" *** DEBUG: draw \"best\" and averaged circle")
        test_img = image.copy()
        print("best x: ", x_best, " best y: ", y_best, " best r: ", r_best)
        cv2.circle(test_img, (x_best, y_best), r_best, (255, 255, 255), 1)
        cv2.circle(test_img, (x_best, y_best), 2, (255, 255, 255), 3)
        print("x_avg: ", x_avg, " y_avg: ", y_avg, " r_avg: ", r_avg)
        cv2.circle(test_img, (x_avg, y_avg), r_avg, (255, 255, 255), 2)
        cv2.circle(test_img, (x_avg, y_avg), 2, (255, 255, 255), 3)
        cv2.imshow("averaged circle with edge image", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #---------------------------------------------------------------------------

    timestamp = time.time()

    return x_best, y_best, r_best, timestamp


def getBallDist(x, y, r, h, w):
    # get distance
    dist_in = (TB_KNOWN_W * CAM_FOCAL_L) / (r*2)
    dist_ft = dist_in / 12
    dist = dist_ft / FT_TO_M_CONV_FACTOR

    # get angle from (0,0) in center of image
    x_map = x - w/2
    y_map = y - h/2
    euc_dist = math.hypot(x_map, y_map)
    euc_dist = euc_dist / M_TO_PX_CONV_FACTOR
    ang = math.asin(euc_dist / dist)

    # DEBUG
    #---------------------------------------------------------------------------
    if debug:
        print(" *** DEBUG: check distance and angle")
        print("x_map: ", x_map, ", y_map: ", y_map)
        print("euc_dist: ", euc_dist)
        print("Distance to Tennis Ball [m]: ", dist)
        print("Angle of Tennis Ball [rad]: ", ang)
    #---------------------------------------------------------------------------

    return dist, ang


def checkCircle(orig_hsv_image, hsv_image, x_start, y_start, r_avg, thresh, h, w):
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
    """
    # DEBUG
    if advanced_debug:
        print(" *** DEBUG: create temp data structure")
        temp = []

    # format: [H, S, V]
    color_accept_low = [25,50,0]
    color_accept_high = [75,255,255]
    d = deque()
    circle_pts = 0
    color_range_pts = 0

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
        if ((info[0] > 0) and (info[0] < (w - 1)) and (info[1] > 0) and (info[1] < (h - 1))):
            length = math.hypot(x_start - info[0], y_start - info[1])
            
            # DEBUG
            if advanced_debug:
                print(" *** DEBUG: check iterated pts")
                print("x: ", info[0], ", y: ", info[1], ", length: ", length)

            if length < r_avg:
                # DEBUG
                if advanced_debug:
                    print(" *** DEBUG: add pts to temp data structure")
                    temp.append([info[0], info[1]])

                # increment total pts in circle counter
                circle_pts += 1;

                # check if pxl values in range
                pxl = orig_hsv_image[info[1], info[0]]
                if ((pxl[0] > color_accept_low[0] and pxl[0] < color_accept_high[0]) and
                    (pxl[1] > color_accept_low[1] and pxl[1] < color_accept_high[1]) and 
                    (pxl[2] > color_accept_low[2] and pxl[2] < color_accept_high[2])):
                    # increment pts in color range counter
                    color_range_pts += 1;

                # continue to next pt(s)
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

    #DEBUG: show checked coordinates and other information
    #---------------------------------------------------------------------------
    if debug:
        print(" *** DEBUG: check iterated pts")
        if advanced_debug:
            test_img = hsv_image.copy()
            for i in temp:
                cv2.circle(test_img, (i[0], i[1]), 1, (255, 255, 255), 1)
                cv2.imshow("averaged circle with edge image", test_img)
                cv2.waitKey(1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("x start: ", x_start, ", y start: ", y_start)
        print("# pts in color range: ", color_range_pts, ", # pts in circle: ", circle_pts, ", fraction of points in acceptable range: ", (color_range_pts / circle_pts))
    #---------------------------------------------------------------------------

    # check if percentage of in-range color pts is above acceptable threshold
    if (color_range_pts / circle_pts) > thresh:
        return True

    return False


def main():
    # setup
    # TODO: replace below (i.e. between ***) with setting up ROS/topic functionality
    # ***************************************************************************
    image_buf = []
    for idx, f in enumerate(os.listdir(temp_path)):
        if idx > 100:
            break
        filepath = os.path.join(temp_path, f)
        image_buf.append(filepath)
    num_found = 0
    num_not_found = 0
    # ***************************************************************************

    first = True
    global frame_stack
    start_time = time.time()

    # main loop
    while (True):
        # read and/or load next frame
        # TODO: replace below (i.e. between ***) with getting next image from topic
        # ***************************************************************************
        if not image_buf:
            break
        next_img= cv2.imread(image_buf.pop())
        # ***************************************************************************

        if len(frame_stack) >= MAX_STACK_SIZE:
            # DEBUG
            if debug:
                print("purging stack")
            frame_stack = []
        else:
            frame_stack.append(next_img)

            t_diff = time.time() - start_time
            # DEBUG
            if debug:
                print("\ntime calc", t_diff, ", ref: ", (1/15))
            if (t_diff > (1/15)) or first:
                first = False
                start_time = time.time()

                print("reading next image")
                img = frame_stack.pop()

                # confirm image w/ cnn
                detected = getNetDetections(img)

                if detected:
                    # pre-processing
                    height, width = img.shape[:2]

                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_img = cv2.medianBlur(img, KERNEL_SIZE)

                    # DEBUG
                    #---------------------------------------------------------------------------
                    if debug:
                        print(" *** DEBUG: initial check")
                        print(str(width) + "x" + str(height))
                        if image_debug:
                            cv2.imshow('image', img)
                            cv2.imshow('gray image', gray_img)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    #---------------------------------------------------------------------------

                    # processing
                    #  - color
                    orig_hsv_img, hsv_img = getHsvMask(blur_img)

                    # get edge image
                    edge_img = cv2.Canny(hsv_img, 0, 255, apertureSize=3, L2gradient=False)

                    # DEBUG: check edge image
                    #---------------------------------------------------------------------------
                    if image_debug:
                        print(" *** DEBUG: display edge img")
                        cv2.imshow('edge img', edge_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        #exit()
                    #---------------------------------------------------------------------------

                    #  - circles
                    x_c, y_c, r, tstamp = getBallLoc(edge_img)
                    if (0 == x_c) and (0 == y_c) and (0 == r) and (0 == tstamp):
                        num_not_found += 1
                        continue

                    # DEBUG
                    #---------------------------------------------------------------------------
                    if debug:
                        print(" *** DEBUG: check detected circle info")
                        if advanced_debug:
                            for h in range((y_c-r),(y_c+r)):
                                for w in range((x_c-r),(x_c+r)):
                                    print(hsv_img[h][w])
                                    #cv2.circle(hsv_img, (w, h), 1, (255, 255, 255), 1)
                                    cv2.imshow("test", hsv_img)
                                    cv2.waitKey(1)
                                cv2.waitKey(0)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        print("x_c: ", x_c, ", y_c: ", y_c, ", r: ", r, ", tstamp: ", tstamp)
                        #exit()
                    #---------------------------------------------------------------------------

                    #  - determine whether to keep or not
                    found = checkCircle(orig_hsv_img, hsv_img, x_c, y_c, r, .7, height, width)

                    # DEBUG
                    if debug:
                        print(" *** DEBUG: check if acceptable circle found")
                        print("found? ", found)
                        #exit()

                    # results
                    if found:
                        num_found += 1

                        #  - distance to ball
                        dist, angle = getBallDist(x_c, y_c, r, height, width)

                        # output information
                        # output information structure:
                        #   [x center, y center, radius, distance to ball, angle of ball, timestamp]
                        output_info = [x_c, y_c, r, dist, angle, tstamp]
                        # TODO: replace below (i.e. between ***) with posting info to topic
                        # ***************************************************************************
                        print(output_info)
                        # ***************************************************************************
                    else:
                        num_not_found += 1
                else:
                    time.sleep(.05)

    # DEBUG
    if debug:
        print("num found: ", num_found, ", num not found: ", num_not_found)
        print("end of main loop")

    sess.close
    return 0


if __name__ == "__main__":
  main()

import cv2
import numpy as np
import time

# "constants"
FRAME_RATE = 15
MAX_STACK_SIZE = 60
CAM_FOCAL_L = 112.0    # what unit is this?
TB_KNOWN_W = 2.6       # [in]
IMG_W = 1920
IMG_H = 1080

frame_stack = []

# cv2 information
# "constants"
CIRCLE_MIN_R = 2
KERNEL_SIZE = 15

min_HSV = np.array([30,40,110])
max_HSV = np.array([55,150,255])

# return information
x_center_final = None
y_center_final = None
r_final = None
tb_dist_final = None
detect_timestamp = None

def get_hsv_mask():
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # setup mask for image
    # NOTE: from very shallow experimentation, I have found that the Hue range of 33 - 37 gives 
    # a nice range of tennis ball-y colors
    lower_yellow = np.array([29,86,6])
    upper_yellow = np.array([64,255,255])
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


def get_candidate_circle():
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([29,86,6])
    upper_yellow = np.array([64,255,255])
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    
    hsv_img[np.where(mask==0)] = 0
    
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=5)
    
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    edge_img = cv2.Canny(mask, 0, 255)
    
    cv2.imshow('edge img', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get circles using Hough Transform
    height, width = edge_img.shape
    bound = min(height, width)
    separation = int(round(bound/100))
    separation_dist = 2 if separation < 2 else separation
    print("bound: ", bound, " separation: ", separation, " separation distance: ", separation_dist)
    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, 1, separation_dist, param1=255, param2=20, minRadius=int(round(bound/3)), maxRadius=int(round(bound/2)))
    if circles is None:
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
    offsets = []
    search_range = math.ceil(r_avg / 10)

    edge_coords = getEdgeCoords(edge_img, x_avg, y_avg, r_avg, search_range)

#    #DEBUG: show detected coordinates
#    #---------------------------------------------------------------------------
#    for i in edge_coords:
#        print("detected edge x-coordinate: ", i[0], " y-coordinate: ", i[1])
#        cv2.circle(test_edge_img, (i[0], i[1]), 20, (255, 255, 255), 1)
#        cv2.imshow("averaged circle with edge image", test_edge_img)
#        cv2.waitKey(1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    #---------------------------------------------------------------------------

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
    goodness = offsets_avg / max_offset

    #DEBUG: check goodness and relevant values
    #---------------------------------------------------------------------------
    print("max offset: ", max_offset, " offsets_tot: ", offsets_tot, " offsets_avg: ", offsets_avg)
    print("goodness: ", goodness)
    #---------------------------------------------------------------------------
    
    return goodness


def find_ball():



def get_ball_loc():



def get_ball_dist():
    res = [x_res, y_res]
    cam_w_actual = (ymax - ymin) * y_res
    dist_in = (KNOWN_WIDTH * FOCAL_LENGTH) / cam_w_actual
    dist_ft = dist_in / 12
    dist = dist_ft / FT_TO_M_CONV_FACTOR

    # DEBUG
    #print("Distance to Tennis Ball [m]: ", dist)


def main():
    # see AI-ROCKS/Drive/Models/Camera.cs for references and structure
    # setup
    # ---------------------------------------------------------------------------
    # NOTE: replace <placeholder> code with what will be used for the image input
    # <placeholder>
    img_src = cv2.imread("./tb_test_images/tennis_ball_1042.jpg")

    start_time = time.time()

    # main loop
    while (True):
        # read and/or load next frame
        # ---------------------------------------------------------------------------
        # NOTE: replace <placeholder> code with what will be used for the image input
        # <placeholder>
        next_img = img_src

        if len(frame_stack) >= MAX_STACK_SIZE:
            frame_stack = []
        else:
            frame_stack.append(next_img)

            if ((time.time() - start_time) % (1 / FRAME_RATE)) < 5:
                img = frame_stack.pop()

            # pre-processing
            # ---------------------------------------------------------------------------
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.medianBlur(img, 5)
            blur_gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

            # with blur (i.e. 11)
            #edge_img = cv2.Canny(blur_gray_img, 10, 37)
            # without blur/with little blur (i.e. 5)
            edge_img = cv2.Canny(blur_gray_img, 37, 47)

            height, width = img.shape[:2]

            #DEBUG
            #---------------------------------------------------------------------------
            print(str(height) + "x" + str(width))
            cv2.imshow('image', img)
            cv2.imshow('gray image', gray_img)
            cv2.imshow('blurred image', blur_img)
            cv2.imshow('gray blurred image', blur_gray_img)
            cv2.imshow('edge image', edge_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
            #---------------------------------------------------------------------------

            # processing
            # ---------------------------------------------------------------------------
            # TODO: make a decision on this
            # use one of 2 methods:
            #   1. SIFT
            #   2. combination of feature detection (circles, color, contrast, etc.)
            # Method 1:
            #  - <SIFT method details>


            # Method 2: 
            #  - color
            get_hsv_mask()

            #  - circles
            get_candidate_circle()

            # results
            # ---------------------------------------------------------------------------
            #  - location in frame
            get_ball_loc()

            #  - distance to ball
            get_ball_dist()

            # output information
            # ---------------------------------------------------------------------------


if __name__ == "__main__":
  main()

import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import PIL
import glob
from sklearn.datasets import load_digits
import seaborn as sns
import pandas as pd
import cv2 as cv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip

# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

folder = "images-56fa467-abs-keepRatioXY-invertY"
folder_rot = "images-56fa467-abs-keepRatioXY-invertY-rotated_x00_y00_z20"
grid_size = 256
img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{folder}\rgb\{grid_size}x{grid_size}\img"
img_real_paths = sorted(glob.glob(os.path.join(img_real_dir, "*.png")))

img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{folder_rot}\rgb\{grid_size}x{grid_size}\img"
img_real_rot_paths = sorted(glob.glob(os.path.join(img_real_rot_dir, "*.png")))

def get_properties(img, thresh = 0, max_value = 255, contour_method = cv.RETR_LIST, plt_bool = False):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked

    thresh_mode = cv.THRESH_BINARY
    ret,img_thresh = cv.threshold(imgray, thresh, max_value, thresh_mode)
    
    # Get contours using RETR_TREE method: retrieves all the contours, creates a complete hierarchy
    # CHAIN_APPROX_NONE --> all contour points will be stored

    contours, hierarchy = cv.findContours(img_thresh, contour_method, cv.CHAIN_APPROX_NONE)

    cnt = contours[0]
    draw_inplace = True
    M = cv.moments(cnt)
    # print( M )

    cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,255,0), thickness=1)

    # Rectangle
    area = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    if draw_inplace:
        cv.drawContours(img,[box],0,(0,0,255),2)

    # # Circle
    # (x,y),radius = cv.minEnclosingCircle(cnt)
    # center = (int(x),int(y))
    # radius = int(radius)
    # if draw_inplace:
    #     cv.circle(img,center,radius,(0,255,0),2)

    # Line
    # rows,cols = img.shape[:2]
    # [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((cols-x)*vy/vx)+y)
    # cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)


    # # Aspect Ratio
    # x,y,w,h = cv.boundingRect(cnt)
    # aspect_ratio = float(w)/h
    # print(f"{aspect_ratio = }")

    # # Extent
    # area = cv.contourArea(cnt)
    # x,y,w,h = cv.boundingRect(cnt)
    # rect_area = w*h
    # extent = float(area)/rect_area
    # print(f"{extent = }")

    # # Solidity
    # area = cv.contourArea(cnt)
    # hull = cv.convexHull(cnt)
    # hull_area = cv.contourArea(hull)
    # solidity = float(area)/hull_area
    # print(f"{solidity = }")


    # # Equivalent Diameter 
    # area = cv.contourArea(cnt)
    # equi_diameter = np.sqrt(4*area/np.pi)
    # print(f"{equi_diameter = }")

    # Orientation
    # (x,y),(MA,ma),angle = cv.fitEllipseAMS(cnt)
    (x,y),(MA,ma),angle = cv.minAreaRect(cnt)

    print(f"{angle = }")
    print(f"{MA = }")
    print(f"{ma = }")
    
    # Mask and pixel points
    mask = np.zeros(imgray.shape,np.uint8)
    cv.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv.findNonZero(mask)

    img[int(y)-2:int(y)+2,int(x)-2:int(x)+2] = [0,0,0]

    # Mean color or mean intensity
    mean_val = cv.mean(imgray,mask = mask)
    
    # print([y,x])
    # print(imgray.shape)
    # # conts = cv.drawContours(img, [cnt], 0, (0,255,0), 3)

    if plt_bool:
        cv.imshow("Minimum Area Rectangle", img)
        cv.imshow('Binary image', img_thresh)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return angle, x, y

angle_list = []

img_num = 0

get_properties(cv.imread(img_real_rot_paths[img_num]).copy())

# img_real_paths = [img_real_paths[img_num]]

for img_real_path in img_real_paths:
    # img_real_path = img_real_paths[0]
    # Reading the image stored as BGR
    image_orig = cv.imread(img_real_path)
    print("\n\nOriginal")
    angle_calc, x, y = get_properties(image_orig.copy())
    image = image_orig
    angle_list.append(angle_calc)
    num_iter = 10

    print(np.array(angle_list).max(), np.array(angle_list).min())

    for ctr in range(num_iter):
        if angle_calc <= 45:
            angle_rot = -angle_calc
        elif angle_calc > 45:
            angle_rot = angle_calc-90

        # angle_rot = float(input("AngleRot: "))
        height, width = image.shape[:2]
        # # get the center coordinates of the image to create the 2D rotation matrix
        # center = (width/2, height/2)
        center = (x, y)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        # rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle_calc-90, scale=1)
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle_rot, scale=1)
        # rotate the image using cv2.warpAffine
        rotated_image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

        print("\n\nRotated")
        angle_calc_rot, _, _ = get_properties(rotated_image.copy())

        if angle_calc_rot <= 45:
            eps = np.abs(angle_calc_rot)
        elif angle_calc > 45:
            eps = np.abs(angle_calc_rot-90)
        
        # cv.imshow('Original image', image_orig)
        # cv.imshow('Rotated image', rotated_image)
        # cv.waitKey(0)

        if eps < 0.01:
            print(f"{eps = }")
            print(f"After {ctr+1} rotations: {os.path.basename(img_real_path)}")
            cv.imshow('Original image', image_orig)
            cv.imshow('Rotated image', rotated_image)
            cv.waitKey(500)
            # save the rotated image to disk
            # cv.imwrite('rotated_image.png', rotated_image)
            break
        elif ctr == num_iter-1:
            print("\n\nFailed\n\n")

        angle_calc = angle_calc_rot
        image = rotated_image

    # # save the rotated image to disk
    # cv.imwrite('rotated_image.jpg', rotated_image)


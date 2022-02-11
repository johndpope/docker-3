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
from hashlib import sha256
import copy

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip


# https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

folder = "images-2ad6e8e-abs-keepRatioXY-invertY-rot_bb"

folder_rot = "images-56fa467-abs-keepRatioXY-invertY-rotated_x00_y00_z20"
grid_size = 256
img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{folder}\rgb\{grid_size}x{grid_size}\img"


img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{folder_rot}\rgb\{grid_size}x{grid_size}\img"
img_real_rot_paths = sorted(glob.glob(os.path.join(img_real_rot_dir, "*.png")))


# Folders for the used network
p_folder = "220208_ffhq-res256-mirror-paper256-noaug"
kimg = "kimg3000"
results_cfg = "00000-img_prep-mirror-paper256-kimg3000-ada-target0.7-bgcfnc-nocmethod-resumecustom-freezed1"
snapshot = "network-snapshot-001392"

# Paths
if os.name == "nt":
    img_dir_base = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"
elif os.name == "posix":
    img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
    img_real_dir = img_dir
    
    img_rot_dir_base = img_dir_base

img_fake_dir = os.path.join(img_dir_base, "images-generated", f"{grid_size}x{grid_size}", p_folder, kimg, results_cfg, snapshot, "img")

# Network hash (needed for caching the feature data)
fake_hash = sha256((p_folder+kimg+results_cfg+snapshot).encode()).hexdigest()[::10]

def get_properties(img, thresh = 20, max_value = 255, contour_method = cv.RETR_LIST, plt_bool = False, print_bool=False, img_path=[], img_new_dir=[]):

    # Check for
    if not img_path == img_new_dir:
        if any([not img_path, not img_new_dir]):
            raise ValueError("Please provice img_path and img_new_dir.")


    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked
    thickness = 1
    thresh_mode = cv.THRESH_BINARY
    ret,img_thresh = cv.threshold(imgray, thresh, max_value, thresh_mode)
    
    if img_path and img_new_dir:
        cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path).replace(".", "_thresh.")), img_thresh)
    # Get contours using RETR_TREE method: retrieves all the contours, creates a complete hierarchy
    # CHAIN_APPROX_NONE --> all contour points will be stored

    contours, hierarchy = cv.findContours(img_thresh, contour_method, cv.CHAIN_APPROX_NONE)

    cnt = contours[0]
    M = cv.moments(cnt)
    # print( M )

    if img_path and img_new_dir:
        contour_img = copy.deepcopy(img)
        cv.drawContours(image=contour_img, contours=contours, contourIdx=-1, color=(0,255,0), thickness=thickness)
        cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path).replace(".", "_contour.")), contour_img)

    # Rectangle
    area = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    if img_path and img_new_dir:
        rect_img = copy.deepcopy(img) 
        cv.drawContours(rect_img,[box],0,(0,0,255),thickness)
        cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path).replace(".", "_rect.")), rect_img)

    # Circle
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)

    if img_path and img_new_dir:
        circle_img = copy.deepcopy(img)
        cv.circle(circle_img,center,radius,(0,255,0),thickness)
        cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path).replace(".", "_circle.")), circle_img)

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
    #(x,y),(MA,ma),angle = cv.fitEllipseAMS(cnt)
    
    (x,y),(MA,ma),angle = cv.minAreaRect(cnt)
    # _,_, (MA,ma),_ = cv.fitEllipseAMS(cnt)

    if print_bool:
        print(f"{angle = }")
        print(f"{MA = }")
        print(f"{ma = }")
    
    # # Mask and pixel points
    # mask = np.zeros(imgray.shape,np.uint8)
    # cv.drawContours(mask,[cnt],0,255,-1)
    # pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv.findNonZero(mask)

    # img[int(y)-2:int(y)+2,int(x)-2:int(x)+2] = [0,0,0]

    # Mean color or mean intensity
    # mean_val = cv.mean(imgray,mask = mask)
    
    # print([y,x])
    # print(imgray.shape)
    # # conts = cv.drawContours(img, [cnt], 0, (0,255,0), 3)

    if plt_bool:
        cv.imshow("Minimum Area Rectangle", img)
        cv.imshow('Binary image', img_thresh)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return angle, x, y



# img_num = 0

# get_properties(cv.imread(img_real_rot_paths[img_num]).copy())

# # img_real_paths = [img_real_paths[img_num]]

def rotate_img(img_dir):
    img_new_dir = img_dir + "-cvRot"
    os.makedirs(img_new_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    angle_list = []
    for img_path in img_paths:
        # img_real_path = img_real_paths[0]
        # Reading the image stored as BGR
        image_orig = cv.imread(img_path)
        print("\n\nOriginal")
        # angle_calc, x, y = get_properties(image_orig.copy(), print_bool=True, plt_bool=False, img_new_dir=img_new_dir, img_path=img_path)
        angle_calc, x, y = get_properties(image_orig.copy(), print_bool=True, plt_bool=False)

        image = image_orig
        angle_list.append(angle_calc)
        num_iter = 100

        for ctr in range(num_iter):
            if angle_calc < 45:
                angle_rot = angle_calc
            elif angle_calc >= 45:
                angle_rot = angle_calc-90

            # angle_rot = float(input("AngleRot: "))
            height, width = image.shape[:2]
            # # get the center coordinates of the rectangle to create the 2D rotation matrix
            center = (x, y)
            # using cv2.getRotationMatrix2D() to get the rotation matrix
            rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle_rot, scale=1)
            # rotate the image using cv2.warpAffine
            rotated_image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

            
            angle_calc_rot, _, _ = get_properties(rotated_image.copy())

            if angle_calc_rot < 45:
                eps = np.abs(angle_calc_rot)
            elif angle_calc_rot >= 45:
                eps = np.abs(angle_calc_rot-90)

            if eps < 0.01:
                print("\n\nRotated")
                print(f"{angle_calc_rot = }")
                print(f"{eps = }")
                print(f"After {ctr+1} rotations: {os.path.basename(img_path)}")
                
                if 1:
                    cv.imshow('Original image', image_orig)
                    cv.imshow('Rotated image', rotated_image)
                    cv.waitKey(500)
                    while 1:
                        if int(input("1/0")):
                            rotate_matrix = cv.getRotationMatrix2D(center=center, angle=90, scale=1)
                            # rotate the image using cv2.warpAffine
                            rotated_image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
                            cv.imshow('Rotated image', rotated_image)
                            cv.waitKey(1000)
                            image = rotated_image
                        else:
                            break

                # cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path)), image_orig)
                # cv.imwrite(os.path.join(img_new_dir, os.path.basename(img_path).replace(".", "_rot.")), rotated_image)
                cv.imwrite(img_path.replace(folder, folder+"-cvRot"), rotated_image)

                break
            elif ctr == num_iter-1:
                print("\n\nFailed\n\n")

            angle_calc = angle_calc_rot
            image = rotated_image

        

        # # save the rotated image to disk
        # cv.imwrite('rotated_image.jpg', rotated_image)

# rotate_img(img_fake_dir)
rotate_img(img_real_dir)

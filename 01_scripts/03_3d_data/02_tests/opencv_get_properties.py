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
grid_size = 256
img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{folder}\rgb\{grid_size}x{grid_size}\img"
img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))

def get_properties(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked

    ret,thresh = cv.threshold(imgray, 3, 255, 0)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    draw_inplace = True
    M = cv.moments(cnt)
    # print( M )

    cv.drawContours(img, contours, -1, (0,255,0), 3)

    # Rectangle
    area = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    if draw_inplace:
        cv.drawContours(img,[box],0,(0,0,255),2)

    # Circle
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    if draw_inplace:
        cv.circle(img,center,radius,(0,255,0),2)

    # Line
    # rows,cols = img.shape[:2]
    # [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((cols-x)*vy/vx)+y)
    # cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)


    # Aspect Ratio
    x,y,w,h = cv.boundingRect(cnt)
    aspect_ratio = float(w)/h
    print(f"{aspect_ratio = }")

    # Extent
    area = cv.contourArea(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    print(f"{extent = }")

    # Solidity
    area = cv.contourArea(cnt)
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    solidity = float(area)/hull_area
    print(f"{solidity = }")


    # Equivalent Diameter 
    area = cv.contourArea(cnt)
    equi_diameter = np.sqrt(4*area/np.pi)
    print(f"{equi_diameter = }")

    # Orientation
    (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    print(f"{angle = }")

    # Mask and pixel points
    mask = np.zeros(imgray.shape,np.uint8)
    cv.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    #pixelpoints = cv.findNonZero(mask)


    # Mean color or mean intensity
    mean_val = cv.mean(imgray,mask = mask)
    img[int(y)-2:int(y)+2,int(x)-2:int(x)+2] = [0,0,0]
    # print([y,x])
    # print(imgray.shape)
    # # conts = cv.drawContours(img, [cnt], 0, (0,255,0), 3)

    cv.imshow("Minimum Area Rectangle", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return angle, x, y


# Reading the image stored as BGR
image = cv.imread(img_real_paths[38])
angle_calc, x, y = get_properties(image.copy())

height, width = image.shape[:2]

# # get the center coordinates of the image to create the 2D rotation matrix
# center = (width/2, height/2)
center = (y, x)

# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle_calc-90, scale=1)

# rotate the image using cv2.warpAffine
rotated_image = cv.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

angle_calc_rot, x_rot, y_rot = get_properties(rotated_image.copy())


cv.imshow('Original image', image)
cv.imshow('Rotated image', rotated_image)
# wait indefinitely, press any key on keyboard to exit
cv.waitKey(0)
# # save the rotated image to disk
# # cv.imwrite('rotated_image.jpg', rotated_image)


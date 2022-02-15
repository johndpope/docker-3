import cv2
import numpy as np
from tqdm import tqdm
import cv2 as cv
import os
import copy


def matchfinder_bf(img1_path, img2_path):
    img_r = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img_rr = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()
    orb.setEdgeThreshold(0)
    orb.setMaxFeatures(10000)
    orb.setNLevels(20)

    kp_r, des_r = orb.detectAndCompute(img_r, None)
    kp_rr, des_rr = orb.detectAndCompute(img_rr, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des_r, des_rr)
    matches = sorted(matches, key=lambda x: x.distance)

    # matches = [m for m in matches if m.distance <= 10]
    match_dis = [m.distance for m in matches]

    if not match_dis:
        match_dis = 100

    return match_dis, matches


class ImageProps:

    contour_mode = cv.RETR_LIST
    contour_method = cv.CHAIN_APPROX_NONE
    thresh = 20 
    thresh_mode = cv.THRESH_BINARY
    default_contour_color = (0,255,0)
    max_value = 255
    thickness = 1
    img_new_dir=[]
    avail_img_types = []


    def __init__(self, img_path):

        self.img_path = img_path
        self.img_orig = cv.imread(self.img_path)
        self.img = self.img_orig
        self.avail_img_types.append("orig")
        self.height, self.width = self.img_orig.shape[:2]
        self.get_contours()
    

    def get_contours(self):
    
        self.img_gray = cv.cvtColor(src = self.img, code = cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked
        
        _, self.img_thresh = cv.threshold(src = self.img_gray, thresh = self.thresh, maxval = self.max_value, type = self.thresh_mode)
        
        # Get contours using RETR_TREE method: retrieves all the contours, creates a complete hierarchy
        # CHAIN_APPROX_NONE --> all contour points will be stored
        self.contours, _ = cv.findContours(image = self.img_thresh, mode = self.contour_mode, method = self.contour_method)

        if "thresh" not in self.avail_img_types:
            self.avail_img_types.append("thresh") 

        self.img_contour = copy.deepcopy(self.img)
        cv.drawContours(image=self.img_contour, contours=self.contours, contourIdx=-1, color=(0,255,0), thickness=self.thickness)
        if "contour" not in self.avail_img_types:
            self.avail_img_types.append("contour")

        self.cnt = self.contours[0]
        M = cv.moments(self.cnt)
        

    def get_rect(self, color=[]):
        if not color:
            color = self.default_contour_color

        self.avail_img_types.append("rect")
        self.rect = cv.minAreaRect(self.cnt)
        box = np.int0(cv.boxPoints(self.rect))
        self.img_rect = copy.deepcopy(self.img) 
        cv.drawContours(image=self.img_rect, contours=[box], contourIdx=-1, color=color, thickness=self.thickness)


    def get_circle(self, color=[]):
        if not color:
            color = self.default_contour_color

        if "circle" not in self.avail_img_types:
            self.avail_img_types.append("circle") 

        self.circle = {}
        (x,y),radius = cv.minEnclosingCircle(self.cnt)
        self.circle["center"]= (int(x),int(y))
        self.circle["radius"] = int(radius)
        self.img_circle = copy.deepcopy(self.img) 
        cv.circle(img = self.img_circle,center = self.circle["center"],radius = self.circle["radius"], color=color, thickness = self.thickness)

    def get_aspect_ratio(self):
        # Aspect Ratio
        x,y,w,h = cv.boundingRect(self.cnt)
        self.aspect_ratio = float(w)/h
        return self.aspect_ratio

    def get_solidity(self):
        area = cv.contourArea(self.cnt)
        hull = cv.convexHull(self.cnt)
        hull_area = cv.contourArea(hull)
        self.solidity = float(area)/hull_area
        return self.solidity

    def get_extent(self):
        area = cv.contourArea(self.cnt)
        x,y,w,h = cv.boundingRect(self.cnt)
        rect_area = w*h
        self.extent = float(area)/rect_area
        return self.extent

    def get_orientation(self):
        self.get_contours()
        self.center,(_,_), self.angle = cv.minAreaRect(self.cnt)
        return self.angle


    def set_orientation_zero(self, eps_max = 0.01, num_iter=100):

        self.img_rot = copy.deepcopy(self.img) 
        self.get_orientation()
        print(f"\nOriginal: \n{self.angle = }")

        for ctr in range(num_iter):
            self.get_orientation()

            if self.angle < 45:
                angle_rot = self.angle
            elif self.angle >= 45:
                angle_rot = self.angle-90

            eps = np.abs(angle_rot)

            if eps < eps_max:

                print("\n\nRotated:")
                print(f"{self.angle = }")
                print(f"{eps = }")
                print(f"After {ctr} rotations: {os.path.basename(self.img_path)}")
                
                if 1:
                    cv.imshow('Original image', self.img_orig)
                    cv.imshow('Rotated image', self.img)
                    cv.waitKey(500)
                    while 1:
                        if int(input("Rotate 90 degress? (1/0) \n")):
                            rotate_matrix = cv.getRotationMatrix2D(center=self.center, angle=90, scale=1)
                            # rotate the image using cv2.warpAffine
                            self.img = cv.warpAffine(src=self.img, M=rotate_matrix, dsize=(self.width, self.height))
                            cv.imshow('Rotated image', self.img)
                            cv.waitKey(1000)
                        else:
                            break
                    self.get_orientation()

                break
            # using cv2.getRotationMatrix2D() to get the rotation matrix
            rotate_matrix = cv.getRotationMatrix2D(center=self.center, angle=angle_rot, scale=1)
            # rotate the image using cv2.warpAffine
            self.img = cv.warpAffine(src=self.img, M=rotate_matrix, dsize=(self.width, self.height))

        if "rot" not in self.avail_img_types:
            self.avail_img_types.append("rot") 
        self.img_rot = copy.deepcopy(self.img) 


    def save_images(self, img_types):

        if not isinstance(img_types, list):
            img_types = [img_types]

        if not self.img_path == self.img_new_dir:
            if any([not self.img_path, not self.img_new_dir]):
                raise ValueError("Please provide img_path and img_new_dir.")

        if self.img_path and self.img_new_dir:
            os.makedirs(self.img_new_dir, exist_ok=True)

        for img_type in img_types:
            if img_type in self.avail_img_types:
                cv.imwrite(os.path.join(self.img_new_dir, os.path.basename(self.img_path).replace(".", f"_{img_type}.")), self.__dict__[f"img_{img_type}"])
            else:
                raise ValueError(f"Requested Image Type <{img_type}> not available. \nChoose from: {self.avail_img_types}")
            

    @ classmethod
    def set_img_dir(cls, img_new_dir):
        cls.img_new_dir = img_new_dir

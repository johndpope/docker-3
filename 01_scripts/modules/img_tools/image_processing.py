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
    img_new_dir=None
    avail_img_types = []
    suffix = None

    def __init__(self, img = None, img_path = None):

        self.img_path = img_path

        if img_path is not None:
            self.img_orig = cv.imread(self.img_path)
        elif img is not None:
            self.img_orig = img
        else:
            raise ValueError("Specify either img or img_path")

        self.img = self.img_orig
        self.avail_img_types.append("orig")
        self.height, self.width = self.img_orig.shape[:2]
        self.get_contours()
    
    def __repr__(self) -> str:
        repr_str = "Loaded image" if self.img_path is None else os.path.basename(self.img_path)
        return f"ImageProps object with:\nName: {repr_str}, Height: {self.height}, Width: {self.width}, Available Image Types: {self.avail_img_types}"

    def get_contours(self, color=None, overwrite_img = True):
        """
        Calculate contours of self.img

        "thresh" and "contour" will be added to available images
        """
        if color is None:
            color = self.default_contour_color

        self.img_gray = cv.cvtColor(src = self.img, code = cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked
        
        _, img_thresh = cv.threshold(src = self.img_gray, thresh = self.thresh, maxval = self.max_value, type = self.thresh_mode)
        
        if overwrite_img:
            self.img_thresh = img_thresh

        # Get contours using RETR_TREE method: retrieves all the contours, creates a complete hierarchy
        # CHAIN_APPROX_NONE --> all contour points will be stored
        self.contours, _ = cv.findContours(image = img_thresh, mode = self.contour_mode, method = self.contour_method)

        if overwrite_img:
            self.img_contour = copy.deepcopy(self.img)
            cv.drawContours(image=self.img_contour, contours=self.contours, contourIdx=-1, color=color, thickness=self.thickness)

        if "thresh" not in self.avail_img_types:
            self.avail_img_types.append("thresh") 

        if "contour" not in self.avail_img_types:
            self.avail_img_types.append("contour")

        self.cnt = self.contours[0]
        # M = cv.moments(self.cnt)
        

    def get_rect(self, color=None):
        """
        Calculate the minAreaRect

        "rect" will be added to available images
        """
        if color is None:
            color = self.default_contour_color

        self.avail_img_types.append("rect")
        self.rect = cv.minAreaRect(self.cnt)
        box = np.int0(cv.boxPoints(self.rect))
        self.img_rect = copy.deepcopy(self.img) 
        cv.drawContours(image=self.img_rect, contours=[box], contourIdx=-1, color=color, thickness=self.thickness)


    def get_circle(self, color=None):
        """
        Calculate the minEnclosingCircle

        "circle" will be added to available images
        """

        if color is None:
            color = self.default_contour_color

        if "circle" not in self.avail_img_types:
            self.avail_img_types.append("circle") 

        self.circle = {}
        (x,y),radius = cv.minEnclosingCircle(self.cnt)
        self.circle["center"]= (int(x),int(y))
        self.circle["radius"] = int(radius)
        self.img_circle = copy.deepcopy(self.img) 
        cv.circle(img = self.img_circle,center = self.circle["center"],radius = self.circle["radius"], color=color, thickness = self.thickness)

    @property
    def aspect_ratio(self):
        # Aspect Ratio
        x,y,w,h = cv.boundingRect(self.cnt)
        return float(w)/h

    @property
    def olidity(self):
        area = cv.contourArea(self.cnt)
        hull = cv.convexHull(self.cnt)
        hull_area = cv.contourArea(hull)
        return float(area)/hull_area

    @property
    def extent(self):
        area = cv.contourArea(self.cnt)
        x,y,w,h = cv.boundingRect(self.cnt)
        rect_area = w*h
        return float(area)/rect_area

    def get_orientation(self):
        """
        Get the current orientation of self.img

        Contours will be calculated.
        """
        self.get_contours(overwrite_img=False)
        self.center,(_,_), self.angle = cv.minAreaRect(self.cnt)
        return self.angle


    def set_orientation_zero(self, mode="manual", show_img = True, eps_max = 0.01, num_iter=100):
        """
        Sets orientation of self.img to zero and saves the output in self.img_rot

        mode == "manual": Image can be rotated +/- 90 degress after automatic rotation

        mode == "auto": no 90 deg rotation afterwards

        show_img: (True/False) Show the rotated and original image

        eps_max:    max difference between the actual angle and zero

        num_iter:   maximal number of rotation-iterations
        """
        modes = ["auto", "manual"]
        if not mode in modes:
            raise ValueError(f"mode must be in {modes}")

        # Get current orientation 
        self.get_orientation()
        # print(f"\nOriginal: \n{self.angle = }")


        for ctr in range(num_iter):

            if self.angle < 45:
                angle_rot = self.angle
            elif self.angle >= 45:
                angle_rot = self.angle-90

            eps = np.abs(angle_rot)

            if eps < eps_max:
                # print("\n\nRotated:")
                # print(f"{self.angle = }")
                # print(f"{eps = }")
                # print(f"After {ctr} rotations: {os.path.basename(self.img_path)}")
                
                if show_img or mode=="manual":
                    if self.img_path is not None:
                        print(f"\n{os.path.basename(self.img_path)}")
                    else:
                        print(f"\nNumber: {ctr}")

                    cv.imshow('Original image', self.img_orig)
                    cv.imshow('Rotated image', self.img)
                    cv.waitKey(500)

                while mode == "manual":
                    user_input = input("Rotate 90 degress? \n+  clockwise \n-  counter-clockwise \n0  Finished \nInput: ")
                    if user_input in ["+", "-"]:
                        angle = -90 if user_input == "+" else 90
                        # In OpenCV a positive angle is counter-clockwise
                        rotate_matrix = cv.getRotationMatrix2D(center=self.center, angle=angle, scale=1)
                        # rotate the image using cv2.warpAffine
                        self.img = cv.warpAffine(src=self.img, M=rotate_matrix, dsize=(self.width, self.height))
                        cv.imshow('Rotated image', self.img)
                        cv.waitKey(1000)
                    elif user_input == "0":
                        break
                    else:
                        continue

                self.get_orientation()
                self.rot_ctr = ctr
                break

            # using cv2.getRotationMatrix2D() to get the rotation matrix
            rotate_matrix = cv.getRotationMatrix2D(center=self.center, angle=angle_rot, scale=1)
            # rotate the image using cv2.warpAffine
            self.img = cv.warpAffine(src=self.img, M=rotate_matrix, dsize=(self.width, self.height))
            self.get_orientation()

        
        if "rot" not in self.avail_img_types:
            self.avail_img_types.append("rot") 
        self.img_rot = copy.deepcopy(self.img) 


    def save_images(self, img_types, suffix=None):
        """
        Saves the requested img_types from self.avail_img_types
        
        set img_types = "all" to compute all img_types except "rot" and save them

        """
        if suffix is None:
            suffix = self.suffix

        if img_types == "all":
            self.get_contours()
            self.get_circle()
            self.get_rect()  
            img_types = self.avail_img_types    

        if not isinstance(img_types, list):
            img_types = [img_types]

        if "rot" in img_types and not "rot" in self.avail_img_types:
            self.set_orientation_zero(mode="auto", show_img=False)


        if not (self.img_path is None and self.img_new_dir is None):
            if any([self.img_path is None, self.img_new_dir is None]):
                raise ValueError("Please provide img_path and img_new_dir.")

        if self.img_path and self.img_new_dir:
            os.makedirs(self.img_new_dir, exist_ok=True)

        for img_type in img_types:
            if img_type in self.avail_img_types:
                replace_str = f"-{img_type}-{suffix}." if suffix is not None and suffix != img_type else f"-{img_type}."
                cv.imwrite(os.path.join(self.img_new_dir, os.path.basename(self.img_path).replace(".", replace_str)), self.__dict__[f"img_{img_type}"])
            else:
                raise ValueError(f"Requested Image Type <{img_type}> not available. \nChoose from: {self.avail_img_types}")
            

    @ classmethod
    def set_img_dir(cls, img_new_dir):
        cls.img_new_dir = img_new_dir


class ImagePropsOrig(ImageProps):
    avail_img_types = []
    suffix = "orig"

    def __init__(self, img_path):
        super().__init__(img_path=img_path)

class ImagePropsRot(ImageProps):
    avail_img_types = []
    suffix = "rot"

    def __init__(self, img_path, mode="manual", show_img=True):
        super().__init__(img_path=img_path)
        self.set_orientation_zero(mode=mode, show_img=show_img)
        self.get_contours()


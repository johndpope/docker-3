import numpy as np
from sklearn.preprocessing import scale
from tqdm import tqdm
import cv2 as cv
import os
import copy
import glob


def matchfinder_bf(img1_path, img2_path):
    img_r = cv.imread(img1_path, cv.IMREAD_COLOR)
    img_rr = cv.imread(img2_path, cv.IMREAD_COLOR)

    orb = cv.ORB_create()
    orb.setEdgeThreshold(0)
    orb.setMaxFeatures(10000)
    orb.setNLevels(20)

    kp_r, des_r = orb.detectAndCompute(img_r, None)
    kp_rr, des_rr = orb.detectAndCompute(img_rr, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

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
    suffix = None
    num_iter=20 # Number of iterations for center and set_orientation_zero
    

    def __init__(self, img = None, rgb_format = None, img_path = None):
        """
        if img (rgb) is provided rgb_format must be provided from ["RGB", "BGR"]

        BGR: OpenCV

        RGB: Pillow

        Internal images arr all BGR format. If RGB or L is needed use the self.export() method.

        """    
        self.avail_img_types = []
        self.error_msg = []

        self.img_path = img_path
        self.img_basename = None
        self.rgb_format = rgb_format

        if img_path is not None:
            self.img_orig = cv.imread(self.img_path)    
            self.img_basename = os.path.basename(self.img_path)
        elif img is not None:      
            channels = np.asarray(img).shape[2] if np.asarray(img).ndim == 3 else 1     
            if self.rgb_format == "RGB" and channels == 3:
                self.img_orig = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            elif self.rgb_format == "BGR" or channels == 1:
                self.img_orig = img
            else:
                raise ValueError('Please provice rgb_format from ["RGB", "BGR"].\nBGR: OpenCV \nRGB: Pillow')

        else:
            raise ValueError("Specify either img or img_path")

        self.img_current = self.img_orig
        self.add_img_type("orig")
        self.add_img_type("current")

        self.height, self.width = self.img_orig.shape[:2]
        self.get_contours()
    
    def __repr__(self) -> str:
        repr_str = "Loaded image" if self.img_path is None else os.path.basename(self.img_path)
        return f"ImageProps object with:\nName: {repr_str}, Height: {self.height}, Width: {self.width}, Available Image Types: {self.avail_img_types}"

    @property
    def contour_area(self):
        return cv.contourArea(self.cnt)

    @property
    def rect_area(self):
        _,_,w,h = cv.boundingRect(self.cnt)
        return w*h

    @property
    def rect_dims(self):
        _,_,w,h = cv.boundingRect(self.cnt)
        return w,h

    @property
    def aspect_ratio(self):
        # Aspect Ratio
        x,y,w,h = cv.boundingRect(self.cnt)
        return float(w)/h

    @property
    def solidity(self):
        area = cv.contourArea(self.cnt)
        hull = cv.convexHull(self.cnt)
        hull_area = cv.contourArea(hull)
        return float(area)/hull_area

    @property
    def extent(self):
        return float(self.contour_area)/self.rect_area


    def add_img_type(self, name):
        """
        Adds name to avail_img_types (per instance) if not already present
        """
        if name not in self.avail_img_types:
            self.avail_img_types.append(name)

    def add_error_msg(self, error_msg):
        """
        Adds err to avail_img_types (per instance) if not already present
        """
        if error_msg not in self.error_msg:
            self.error_msg.append(error_msg)



    def get_contours(self, color=None, overwrite_img = True):
        """
        Calculate contours of self.img

        "thresh" and "contour" will be added to available images
        """
        if color is None:
            color = self.default_contour_color

        # Handle L or RGB images, change code to BGR if img_path is provided or RGB if img is provided
        # Default opencv RGB load format is BGR
        self.channels = np.asarray(self.img_current).shape[2] if np.asarray(self.img_current).ndim == 3 else 1
        if self.channels == 3:
            self.img_gray = cv.cvtColor(src = self.img_current, code = cv.COLOR_BGR2GRAY)    # conversion to grayscale // checked
        elif self.channels == 1:
            self.img_gray = copy.deepcopy(self.img_current)

        _, img_thresh = cv.threshold(src = self.img_gray, thresh = self.thresh, maxval = self.max_value, type = self.thresh_mode)
        
        if overwrite_img:
            self.img_thresh = img_thresh

        # Get contours using RETR_TREE method: retrieves all the contours, creates a complete hierarchy
        # CHAIN_APPROX_NONE --> all contour points will be stored
        self.contours, _ = cv.findContours(image = img_thresh, mode = self.contour_mode, method = self.contour_method)

        if overwrite_img:
            self.img_contour = copy.deepcopy(self.img_current)
            cv.drawContours(image=self.img_contour, contours=self.contours, contourIdx=-1, color=color, thickness=self.thickness)

        self.add_img_type("thresh")
        self.add_img_type("contour")

        # Get the contour with max number of points
        pixel_pairs = [cnt.shape[0] for cnt in self.contours]
        self.countour_idx = np.argmax(pixel_pairs)
        self.cnt = self.contours[self.countour_idx]

        # M = cv.moments(self.cnt)
        

    def get_rect(self, color=None):
        """
        Calculate the minAreaRect

        "rect" will be added to available images
        """
        if color is None:
            color = self.default_contour_color

        self.add_img_type("rect")
        self.rect = cv.minAreaRect(self.cnt)
        box = np.int0(cv.boxPoints(self.rect))
        self.img_rect = copy.deepcopy(self.img_current) 
        cv.drawContours(image=self.img_rect, contours=[box], contourIdx=-1, color=color, thickness=self.thickness)


    def get_circle(self, color=None):
        """
        Calculate the minEnclosingCircle

        "circle" will be added to available images
        """

        if color is None:
            color = self.default_contour_color


        self.circle = {}
        (x,y),radius = cv.minEnclosingCircle(self.cnt)
        self.circle["center"]= (int(x),int(y))
        self.circle["radius"] = int(radius)
        self.img_circle = copy.deepcopy(self.img_current) 
        self.add_img_type("circle")
        cv.circle(img = self.img_circle, center = self.circle["center"], radius = self.circle["radius"], color=color, thickness = self.thickness)

    def get_orientation(self):
        """
        Get the current orientation of self.img

        Contours will be calculated.
        """
        self.get_contours(overwrite_img=False)
        self.rect_center,(_,_), self.angle = cv.minAreaRect(self.cnt)
        return self.angle

    def scale(self, scale_factor, mode="xy"):
        """
        modes = ["xy", "area"]

        xy: scale_factor = X_scale/X = Y_scale/Y

        area: scale_factor = sqrt(scale_factor) = area_scale/area

        """
        modes = ["xy", "area"]
        if mode is None or mode not in modes:
            raise ValueError(f"Provide mode from {modes}")

        if mode == "area":
            scale_factor = np.sqrt(scale_factor)   
        self.get_orientation()
        scale_mat = cv.getRotationMatrix2D(center=self.rect_center, angle=0, scale=scale_factor)
        self.img_current = cv.warpAffine(src=self.img_current, M = scale_mat, dsize=(self.width, self.height))

        self.img_scale = copy.deepcopy(self.img_current)
        self.add_img_type("scale")

    def center(self, eps_max = 0.1):  
        """
        Puts image content in the center of the image

        Goal: img.minAreaRect.center = (img.width/2, img_height/2)

        eps  is defined as np.abs(trans_x)+np.abs(trans_y)
        """
        for ctr in range(self.num_iter):
            self.get_orientation()
            # Calc the needed translation in x and y
            trans_x = self.width/2.-self.rect_center[0]
            trans_y = self.height/2.-self.rect_center[1]
            # Calc eps as sum(x,y)
            eps = np.abs(trans_x)+np.abs(trans_y)
            if eps <= eps_max:
                break
            elif eps > eps_max:
                # Create AffineTransformation matrix: only translation in x and y 
                # x_new = M11*x + M12*y + M13 = 1*x + 0*y + trans_x
                # y_new = M21*x + M22*y + M23 = 0*x + 1*y + trans_y
                M = np.array([[1, 0, trans_x],[0,1,trans_y]], dtype=np.float64)
                self.img_current = cv.warpAffine(src=self.img_current, M=M, dsize=(self.width, self.height))
                self.get_orientation()

        if eps <= eps_max:
            self.add_img_type("center")
            self.img_center = copy.deepcopy(self.img_current)
        else:
            img_name = self.img_basename if self.img_basename is not None else "Image"
            self.add_error_msg(f"{img_name} could not be centered after {ctr+1} tries. eps = {eps:.3f}")

            
    def set_orientation_zero(self, mode="manual", center=True, show_img = True, eps_max = 0.01):
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

        if center:
            self.center()

        # print(f"\nOriginal: \n{self.angle = }")

        for ctr in range(self.num_iter):

            if self.angle < 45:
                angle_rot = self.angle
            elif self.angle >= 45:
                angle_rot = self.angle-90

            eps = np.abs(angle_rot)

            if eps <= eps_max:
                # print("\n\nRotated:")
                # print(f"{self.angle = }")
                # print(f"{eps = }")
                # print(f"After {ctr} rotations: {os.path.basename(self.img_path)}")
                
                if show_img or mode=="manual":
                    if self.img_path is not None:
                        print(f"\n{os.path.basename(self.img_path)}")

                    cv.imshow('Original image', self.img_orig)
                    cv.imshow('Rotated image', self.img_current)
                    cv.waitKey(500)

                while mode == "manual":
                    user_input = input("Rotate 90 degress? \n+  clockwise \n-  counter-clockwise \n0  Finished \nInput: ")
                    if user_input in ["+", "-"]:
                        angle = -90 if user_input == "+" else 90
                        # In OpenCV a positive angle is counter-clockwise
                        rotate_matrix = cv.getRotationMatrix2D(center=self.rect_center, angle=angle, scale=1)
                        # rotate the image using cv.warpAffine
                        self.img_current = cv.warpAffine(src=self.img_current, M=rotate_matrix, dsize=(self.width, self.height))
                        cv.imshow('Rotated image', self.img_current)
                        cv.waitKey(1000)
                    elif user_input == "0":
                        break
                    else:
                        continue

                self.get_orientation()
                self.rot_ctr = ctr
                break
            
            # using cv.getRotationMatrix2D() to get the rotation matrix
            rotate_matrix = cv.getRotationMatrix2D(center=self.rect_center, angle=angle_rot, scale=1)
            # rotate the image using cv.warpAffine
            self.img_current = cv.warpAffine(src=self.img_current, M=rotate_matrix, dsize=(self.width, self.height))
            self.get_orientation()

        if center:
            self.center()

        if eps <= eps_max:
            self.add_img_type("rot")
            self.img_rot = copy.deepcopy(self.img_current) 
        else:
            img_name = self.img_basename if self.img_basename is not None else "Image"
            self.add_error_msg(f"{img_name} could not be centered after {ctr+1} tries. eps = {eps:.3f}")




    def crop(self, show_img=False):
        """
        Crop images to main contour

        Gets rid of side-artifacts for img to pcd conversion
        """

        self.get_contours(overwrite_img=False)
        
        # Create mask where white is what we want, black otherwise
        img_mask = np.zeros_like(self.img_gray) 
        # Draw filled (thickness=-1) contour in mask
        cv.drawContours(image=img_mask, contours=self.contours, contourIdx=self.countour_idx, color=255, thickness=-1) 
        # Extract out the object and place into output image
        img_crop = np.zeros_like(self.img_gray) 
        img_crop[img_mask == 255] = self.img_gray[img_mask == 255]

        if self.channels == 3:
            img_crop = cv.cvtColor(src=img_crop, code=cv.COLOR_GRAY2BGR)

        self.img_current = copy.deepcopy(img_crop)  
        self.img_crop = copy.deepcopy(img_crop)  
        self.add_img_type("crop")
        self.get_contours(overwrite_img=False)

        if show_img:
            # Show the output image
            cv.imshow('Cropped Image', img_crop)
            cv.imshow('Mask', img_mask)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def export(self, img_type, rgb_format = None):
        """
        Returns img with requested img_type and specified rgb_format

        See available image types in self.avail_img_types

        img_type = "current" is the latest state of the img after all specified transformations

        If rgb_format was specified @ class instance init and no rgb_format is specified then: rgb_format = rgb_format(init)
        
        """

        if rgb_format is None:
            rgb_format = self.rgb_format

        if img_type in self.avail_img_types:
            
            img = self.__dict__[f"img_{img_type}"]
            channels = np.asarray(img).shape[2] if np.asarray(img).ndim == 3 else 1 
            if channels == 3:
                if rgb_format == "RGB":
                    return cv.cvtColor(img, cv.COLOR_BGR2RGB)
                elif rgb_format == "BGR":
                    return img
                elif rgb_format is None:
                    raise ValueError('Please provice rgb_format from ["RGB", "BGR"].\nBGR: OpenCV \nRGB: Pillow')
            elif channels == 1:
                return img
        else:
            raise ValueError(f"Requested Image Type <{img_type}> not available. \nChoose from: {self.avail_img_types}")

    def save_images(self, img_types, img_basename = None, img_type_paths = None, img_new_dir = None, suffix=None):
        """
        Saves the requested img_types from self.avail_img_types
        
        set img_types = "all" to save all available img_types. Computes all img_types except "rot" and "scale" automatically. 

        img_new_dir: Either set it for the whole class via self.set_img_dir(img_new_dir) or directly via function arg

        suffix: img_name = img_basename-img_type-suffix

        """
        if img_new_dir is None:
            img_new_dir = self.img_new_dir
        if img_new_dir is None:
            raise ValueError("Please provide img_new_dir.")

        if img_basename is not None:
            if "." in img_basename:
                self.img_basename = img_basename
            else:
                self.img_basename = img_basename + ".png"
        elif img_basename is None and self.img_basename is None:
            raise ValueError("Please provide either <img (__init__) and img_basename> or img_path")

        if suffix is None:
            suffix = self.suffix

        if img_type_paths is None:
            img_type_paths = {}

        if not isinstance(img_types, list):
            img_types = [img_types]

        if "all" in img_types:
            self.get_contours()
            self.get_circle()
            self.get_rect()  
            img_types = self.avail_img_types    



        if "rot" in img_types and not "rot" in self.avail_img_types:
            self.set_orientation_zero(mode="auto", show_img=False)
        
        if (self.img_path is not None or self.img_basename is not None) and img_new_dir is not None:
            os.makedirs(img_new_dir, exist_ok=True)
        else:
            raise ValueError("Please provide <img_path and img_new_dir> or <img and img_basename and img_new_dir>.")

        for img_type in img_types:
            if img_type in self.avail_img_types:
                if img_type == "current":
                    new_img_name = self.img_basename
                else:
                    replace_str = f"-{img_type}-{suffix}." if suffix is not None and suffix != img_type else f"-{img_type}."
                    new_img_name = self.img_basename.replace(".", replace_str) if not img_type in img_type_paths.keys() else img_type_paths[img_type]

                save_img = self.__dict__[f"img_{img_type}"]
                cv.imwrite(os.path.join(img_new_dir, new_img_name), save_img)
            else:
                raise ValueError(f"Requested Image Type <{img_type}> not available. \nChoose from: {self.avail_img_types}")

    
    @ classmethod
    def set_img_new_dir(cls, img_new_dir):
        """
        Set the Directory for the new images
        """
        cls.img_new_dir = img_new_dir


class ImagePropsOrig(ImageProps):
    avail_img_types = []
    suffix = "orig"

    def __init__(self, img_path):
        """
        if img is provided rgb_format must be provided from ["RGB", "BGR"]

        BGR: OpenCV

        RGB: Pillow

        """  
        super().__init__(img_path=img_path)

class ImagePropsRot(ImageProps):
    avail_img_types = []
    suffix = "rot"

    def __init__(self, img = None, rgb_format = None, center = True, img_path = None, mode="manual", show_img=True):
        """
        if img is provided rgb_format must be provided from ["RGB", "BGR"]

        BGR: OpenCV

        RGB: Pillow

        """  
        super().__init__(img = img, rgb_format = None, img_path=img_path)
        self.set_orientation_zero(mode=mode, center=center, show_img=show_img)
        self.get_contours()

class ImagePropsScale(ImageProps):
    avail_img_types = []
    suffix = "scale"

    def __init__(self, scale_factor, img = None, rgb_format = None, img_path = None, mode="area"):
        """
        if img is provided rgb_format must be provided from ["RGB", "BGR"]

        BGR: OpenCV

        RGB: Pillow

        """  
        super().__init__(img = img, rgb_format = None, img_path=img_path)
        self.scale(scale_factor=scale_factor, mode=mode)
        self.get_contours()


def ImagePostProcessing(img_dir=None, img_path=None, img=None, rgb_format=None, img_basename = None, img_new_dir = None, scale_mode="area", rot_mode="auto", rot=True, center = True, crop=True, scale=True):
    """
    Accepts following inputs:

    img_dir:    Directory with all images if img_new_dir is None:  img_new_dir = os.path.join(img_dir, "rot-scale")

    img_path:   Path to single image if img_new_dir is None:  img_new_dir = os.path.join(os.path.dirname(img_path), "rot-scale")

    img, rgb_format, img_basename, img_new_dir: Image with rgb format ["RGB", "BGR"], name for new image and directory for new image

    """
    if img_dir is not None:
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    elif img_path is not None:
        img_paths = [img_path]
    elif img is not None and img_new_dir is None:
        raise ValueError("Please provide img_new_dir")

    if img_new_dir is None:
        foldername = "rot-scale"
        if img_dir is not None:
            img_new_dir = os.path.join(img_dir, foldername)
        elif img_path is not None:
            img_new_dir = os.path.join(os.path.dirname(img_path), foldername)

    if not os.path.exists(img_new_dir) or len(os.listdir(img_new_dir)) < len(img_paths):
        images = []
        area_list = []

        if img is None:
            for img_path in tqdm(iterable=img_paths,
                    desc=f"Calculating min_area of {len(img_paths)} images..",
                    ascii=False,
                    ncols=100):
                image = ImageProps(img_path=img_path)
                images.append(image)
                area_list.append(image.rect_area)
        else:
            image = ImageProps(img=img, rgb_format=rgb_format)
            images.append(image)
            area_list.append(image.rect_area)

        min_area = np.min(area_list)
        scale_factors = min_area/area_list


        for image, scale_factor in zip(tqdm(iterable=images,
                    desc=f"Scaling and Rotation of {len(img_paths)} images..",
                    ascii=False,
                    ncols=100) , scale_factors):
            if crop:
                image.crop(show_img=False)
            if center:
                image.center(eps_max=0.2)
            if rot:
                image.set_orientation_zero(mode=rot_mode, center=False, show_img=False)
            if scale:
                image.scale(scale_factor=scale_factor, mode=scale_mode)
            if not image.error_msg:
                image.save_images(img_types=["current"], img_new_dir=img_new_dir, img_basename=img_basename)
            else:
                image.save_images(img_types=["current"], img_new_dir=os.path.join(img_new_dir, "error"), img_basename=img_basename)
                print(image.error_msg)
    else:
        print(f"Images already exist at: \n{img_new_dir}")
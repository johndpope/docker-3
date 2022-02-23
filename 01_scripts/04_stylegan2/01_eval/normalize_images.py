import sys
import os
import glob
from tkinter import Image
import cv2 as cv

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip

img_dir =  r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated\256x256\220222_ffhq-res256-mirror-paper256-noaug\kimg0750\00004-img_prep-mirror-paper256-kimg750-ada-target0.5-bgcfnc-bcr-resumecustom-freezed0\network-snapshot-000418\img"
img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))


area_list = [ip.ImageProps(img_path=img_path).rect_area for img_path in img_paths]
print(area_list[:20])


# print(ImageProps.contour_area)
# print(ImageProps.rect_area)
# print(ImageProps.extent)
# print(ImageProps.rect_dims)
# ImageProps.get_rect()

# ImageProps.scale(scale_factor=0.5, mode="area")

# # cv.imshow('Original image', ImageProps.img_orig)
# # cv.imshow('Scale image', ImageProps.img_scale)
# # cv.waitKey(0)

# ImagePropsScale = ip.ImageProps(img=ImageProps.img_scale)
# print(ImagePropsScale.contour_area)
# print(ImagePropsScale.rect_dims)
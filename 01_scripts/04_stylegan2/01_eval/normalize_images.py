import sys
import os
import glob
from tkinter import Image
import cv2 as cv
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip

img_dir =  r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated\256x256\220222_ffhq-res256-mirror-paper256-noaug\kimg0750\00004-img_prep-mirror-paper256-kimg750-ada-target0.5-bgcfnc-bcr-resumecustom-freezed0\network-snapshot-000418\img"
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:50]

ip.ImageProps.set_img_dir(os.path.join(img_dir, "rot_scale"))

area_list = [ip.ImageProps(img_path=img_path).rect_area for img_path in img_paths]

min_area = np.min(area_list)
scale_factors = min_area/area_list

for img_path, scale_factor in zip(img_paths, scale_factors):
    ImagePropsRot = ip.ImagePropsRot(img_path=img_path, mode="auto", show_img=False)
    ImagePropsRot.scale(scale_factor=scale_factor, mode="area")
    ImagePropsRot.save_images(img_types=["scale", "rot", "orig"])

print("Finished.")





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
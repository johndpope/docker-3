import sys
import os
import glob
from tkinter import Image
import numpy as np
import json
import sys


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp


g_dir = r"G:\ukr_data\Einzelzaehne_sorted\grid"

# img_dir = r"C:\Users\bra45451\Downloads\images-7054d07-abs-keepRatioXY-invertY-rot_3d-full-rot_2d\rgb\256x256\img"
# img_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated\256x256\220224_ffhq-res256-mirror-paper256-noaug\kimg0750\00001-img_prep-mirror-paper256-kimg750-ada-target0.5-bgc-nocmethod-resumecustom-freezed0\network-snapshot-000418\img_post\error"
# img_dir = r"C:\Users\bra45451\Downloads\test"
# img_dir = r"C:\Users\bra45451\Desktop\compare"

# dp.ImageConverterParams(img_dir, param_hash = "dc79a37")
# ImageConverterMulti = dp.ImageConverterMulti(crop=False, center=False, rot=False)
# ImageConverterMulti.img_to_pcd_multi(save_pcd=False, visu_bool=True)

img_dir = r"C:\Users\bra45451\Desktop\compare\test_epscenter_1_epsrot_1"
img_paths = glob.glob(os.path.join(img_dir, "*.png"))

for img_path in img_paths:
    param_hash = os.path.basename(img_path).split(".")[0].split("_")[-1]
    dp.ImageConverterParams(param_hash=param_hash)
    ImageConverterMulti = dp.ImageConverterSingle(img_path=img_path, crop=False, center=False, rot=False)
    ImageConverterMulti.img_to_pcd(save_pcd=False, visu_bool=True)
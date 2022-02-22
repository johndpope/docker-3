import sys
import os
import glob
import numpy as np
import json
import sys


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing2 as dp


g_dir = r"G:\ukr_data\Einzelzaehne_sorted\grid"

img_dir = r"C:\Users\bra45451\Downloads\images-7054d07-abs-keepRatioXY-invertY-rot_3d-full-rot_2d\rgb\256x256\img"

dp.ImageConverterParams(img_dir)
dp.ImageConverterMulti().img_to_pcd(pcd_save=True)


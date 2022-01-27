from hashlib import sha256
import numpy as np
import os
import sys
import glob
from PIL import Image

sys.path.append(os.path.abspath("modules"))
import pcd_tools.data_processing as dp

img_g = sorted(
    glob.glob(
        r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\greyscale\128x128\e14c8f1\img\*"
    ))
img_rgb = sorted(
    glob.glob(
        r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\rgb\128x128\e14c8f1\img\*"
    ))

img1_L = np.asarray(Image.open(img_g[0]))
img1_rgb = np.asarray(Image.open(img_rgb[0]))
img1_rgb_to_L = np.asarray(Image.open(img_rgb[0]).convert("L"))
pixel = 32
print(img1_L[pixel, pixel], img1_rgb[pixel, pixel], img1_rgb_to_L[pixel,
                                                                  pixel])

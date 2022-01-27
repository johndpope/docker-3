import sys
import os
import glob
import numpy as np
import json
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))
import pcd_tools.data_processing as dp

# p_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\models\stylegan2\211208_tfl_celebahq_256\results\00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom\projector_out\img_0_6fe3cda"
# img_path = glob.glob(os.path.join(p_dir, "*.png"))[0]
# grid_size = [256, 256]

g_dir = r"G:\ukr_data\Einzelzaehne_sorted\grid"


pcd_arr = dp.img_to_pcd(img_path)
print(pcd_arr.shape, pcd_arr[:10,:])
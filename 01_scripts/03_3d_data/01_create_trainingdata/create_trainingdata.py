import os
import glob
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp

grid_sizes = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]
grid_sizes = [[256, 256], [512, 512]]

rotation_deg_xyz_list = [[0,0,rotation_deg_z] for rotation_deg_z in range(5,46,5)] # Set to [None] if unused, dtype: np.array
# rotation_deg_xyz_list =  [None]

z_threshold = 4
normbounds = [0, 1]
frame_size = 0.1
nan_val = 15
conversion_type = "abs"
invertY =  False 
keep_xy_ratio = False 

# Directories
stl_dir = r"G:\ukr_data\Einzelzaehne_sorted"
pcd_dir = rf"W:\ukr_data\Einzelzaehne_sorted\grid"
img_dir_base = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\real"


for rotation_deg_xyz in rotation_deg_xyz_list:    
    dp.create_trainingdata_full(
        stl_dir=stl_dir,
        rotation_deg_xyz=rotation_deg_xyz,
        invertY=invertY, 
        grid_sizes=grid_sizes,
        z_threshold=z_threshold, 
        normbounds=normbounds, 
        frame_size=frame_size, 
        nan_val=nan_val, 
        conversion_type=conversion_type, 
        keep_xy_ratio=keep_xy_ratio,
        pcd_dir=pcd_dir, 
        img_dir_base=img_dir_base)

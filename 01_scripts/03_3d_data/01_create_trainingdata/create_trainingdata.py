import os
import glob
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp

grid_sizes = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]

grid_sizes = [[256, 256]]

z_threshold = 4
normbounds = [0, 1]
frame_size = 0.2    # Old: =0.1
nan_val = 15
# New Parameters
conversion_type = "abs" # from ["abs", "rel"]
invertY =  True 
keep_xy_ratio = True 
rot_3d = True
rot_3d_mode = "full"   # from ["full", "z", "bb"]
rot_2d = True
rot_2d_mode = "auto"   # from ["auto", "manual"]
rot_2d_show_img = False
rot_2d_center = True
reduced_data_set = True
reduce_num = 5

# Directories
stl_dir = r"G:\ukr_data\Einzelzaehne_sorted"
pcd_dir = r"W:\ukr_data\Einzelzaehne_sorted\grid"
cfg_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
img_dir_base = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"

dp.DataCreatorParams(     z_threshold=z_threshold,
                    normbounds=normbounds,
                    frame_size=frame_size,
                    nan_val=nan_val,
                    stl_dir=stl_dir,
                    pcd_dir=pcd_dir,
                    cfg_dir=cfg_dir,
                    img_dir_base=img_dir_base,
                    conversion_type=conversion_type,
                    invertY=invertY,
                    keep_xy_ratio=keep_xy_ratio,
                    rot_3d=rot_3d,
                    rot_3d_mode=rot_3d_mode,
                    rot_2d=rot_2d,
                    rot_2d_mode=rot_2d_mode,
                    rot_2d_show_img = rot_2d_show_img,
                    rot_2d_center=rot_2d_center,
                    reduced_data_set=reduced_data_set,
                    reduce_num=reduce_num
)

for grid_size in grid_sizes:
    current_dataset = dp.DatasetCreator(grid_size=grid_size)
    current_dataset.create_trainingdata()

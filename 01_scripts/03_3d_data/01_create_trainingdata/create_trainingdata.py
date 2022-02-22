import os
import glob
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing2 as dp
import img_tools.image_processing as ip

grid_sizes = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]

grid_sizes = [[256, 256]]

# rotation_deg_xyz_list = [[0,0,rotation_deg_z] for rotation_deg_z in range(5,46,5)] # Set to [None] if unused, dtype: np.array
rotation_deg_xyz_list =  [None]

z_threshold = 4
normbounds = [0, 1]
frame_size = 0.1
nan_val = 15
conversion_type = "abs"
invertY =  True 
keep_xy_ratio = True 
rot_3d = True
rot_3d_mode = "full"   # from ["full", "z"]
rot_2d = True
rot_2d_mode = "manual"   # from ["auto", "manual"]

# param_sets = [True, False]
# param_rot_sets = [[None]]

# for param_set in param_sets:
#     for param_rot_set in param_rot_sets:
#         invertY = param_set
#         keep_xy_ratio = param_set
#         rotation_deg_xyz_list = param_rot_set

# Directories
stl_dir = r"G:\ukr_data\Einzelzaehne_sorted\plane_rot"
pcd_dir = r"W:\ukr_data\Einzelzaehne_sorted\grid"
cfg_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
img_dir_base = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"

dp.DataParams(     z_threshold=z_threshold,
                    normbounds=normbounds,
                    frame_size=frame_size,
                    nan_val=nan_val,
                    conversion_type=conversion_type,
                    invertY=invertY,
                    keep_xy_ratio=keep_xy_ratio,
                    rot_3d=rot_3d,
                    rot_3d_mode=rot_3d_mode,
                    rot_2d=rot_2d,
                    rot_2d_mode=rot_2d_mode,
                    stl_dir=stl_dir,
                    pcd_dir=pcd_dir,
                    cfg_dir=cfg_dir,
                    img_dir_base=img_dir_base)

for grid_size in grid_sizes:
    current_dataset = dp.Dataset(grid_size=grid_size)
    current_dataset.create_trainingdata()

# stl_path = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))[0]
# tooth = dp.DataFile(stl_path=stl_path)

# print(params.__dict__)

# # for stl_path in sorted(glob.glob(os.path.join(stl_dir, "*.stl"))):
# #     num = int(stl_path.split(".")[0].split("_")[-1])
# #     new_name = os.path.join(os.path.dirname(stl_path), f"einzelzahn_{num:04d}.stl")
# #     os.rename(stl_path, new_name)


# for rotation_deg_xyz in rotation_deg_xyz_list:    
#     directories = dp.create_trainingdata_full(
#         stl_dir=stl_dir,
#         rotation_deg_xyz=rotation_deg_xyz,
#         invertY=invertY, 
#         grid_sizes=grid_sizes,
#         z_threshold=z_threshold, 
#         normbounds=normbounds, 
#         frame_size=frame_size, 
#         nan_val=nan_val, 
#         conversion_type=conversion_type, 
#         keep_xy_ratio=keep_xy_ratio,
#         rot_3d=rot_3d,
#         rot_2d=rot_2d,
#         pcd_dir=pcd_dir, 
#         img_dir_base=img_dir_base)


        


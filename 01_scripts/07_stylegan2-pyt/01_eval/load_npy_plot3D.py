import os
import glob
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp


model_path = r"G:\ukr_data\Einzelzaehne_sorted\grid_framed_abs\256x256"
filename = "einzelzaehne_train_lb0_ub1.npy"
pcd_to_grid_cfg_name = "pcd_to_grid_cfg.npz"

# np_files = sorted(glob.glob(, key=os.path.getmtime)

np_file = np.load(os.path.join(model_path, filename))

# Load pcd_to_grid cfg file
pcd_to_grid_cfg = np.load(os.path.join(model_path, pcd_to_grid_cfg_name))
grid_size = pcd_to_grid_cfg["grid_size"]
z_threshold = pcd_to_grid_cfg["z_threshold"]
expansion_max = pcd_to_grid_cfg["expansion_max"]

dp.np_2D_to_grid_pcd(
    np_file=np_file[0, :, :, :],
    normbounds=[0, 1],
    grid_size=grid_size,
    z_threshold=z_threshold,
    expansion_max=expansion_max,
    z_crop=0,
    visu_bool=True,
)

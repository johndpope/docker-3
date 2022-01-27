import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))
import pcd_tools.data_processing as dp
import dnnlib.util as util

foldername = "211201"
# p_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\models\stylegan2\211208_tfl_celebahq_256\results\00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom\projector_out\img_0_6fe3cda"
p_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\models\stylegan2\211130_2\img_gen"
param_hash = "6fe3cda"

# img_gen_dir = os.path.join(p_dir, foldername, "img_gen")
img_gen_dir = p_dir

for img_path in glob.glob(os.path.join(p_dir, "*.png")):
    print(img_path)
    # np_savepath = os.path.join(img_gen_dir, f"img_gen_{foldername}.npy")
    np_savepath = os.path.join(img_gen_dir, f"{os.path.basename(img_path).split('.')[0]}.npy")

    if not os.path.exists(np_savepath) or util.ask_yes_no(
            "Overwrite existing npy-File?"):
        dp.img_to_2D_np(img_path=img_path,
                        np_savepath=np_savepath,
                        fileextension="png")

    # Load the images from np array
    # print(f"Loading existing np-File {os.path.basename(np_savepath)}..")
    images = np.load(np_savepath)

    pcd_to_grid_cfg_list = glob.glob(
        os.path.join(
            r"G:\ukr_data\Einzelzaehne_sorted\grid_framed_abs",
            f"{images.shape[1]}x{images.shape[1]}",
            f"pcd_to_grid_cfg_{param_hash}.npz",
        ))

    if len(pcd_to_grid_cfg_list) > 1:
        for num, pathname in enumerate(pcd_to_grid_cfg_list):
            print(f"Index {num}: {os.path.basename(pathname)}")
        pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[int(
            input(f"Enter Index for preferred cfg-File: "))]
    else:
        pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[0]

    # Load pcd_to_grid cfg file
    pcd_to_grid_cfg = np.load(pcd_to_grid_cfg_path)

    grid_size = pcd_to_grid_cfg["grid_size"]
    z_threshold = pcd_to_grid_cfg["z_threshold"]
    expansion_max = pcd_to_grid_cfg["expansion_max"]

    dp.np_2D_to_grid_pcd(
        np_filepath=np_savepath,
        normbounds=[0, 1],
        grid_size=grid_size,
        z_threshold=z_threshold,
        expansion_max=expansion_max,
        z_crop=0,
        visu_bool=True,
        save_pcd_bool=True,
    )

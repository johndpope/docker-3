import os
import glob
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp

grid_sizes = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]

grid_sizes = [[256,256]]

for rotation_deg_z in range(5,46,5):
    # Parameters (numpoints in loop)
    rotation_deg_xyz = [0,0,rotation_deg_z] # Set to None if unused, dtype: np.array
    invertY =  False 
    z_threshold = 4
    normbounds = [0, 1]
    frame_size = 0.1
    nan_val = 15

    pcd_dir = r"G:\ukr_data\Einzelzaehne_sorted\grid"
    img_dir_base = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"

    if rotation_deg_xyz is not None:  
        rot_folder = f"rotated_x{rotation_deg_xyz[0]:02d}_y{rotation_deg_xyz[1]:02d}_z{rotation_deg_xyz[2]:02d}"
        pcd_dir = os.path.join(pcd_dir, "rotated", rot_folder)
        img_dir_base = os.path.join(img_dir_base, "rotated", rot_folder)
    
    if invertY:  
        pcd_dir = os.path.join(pcd_dir, "invertY")
        img_dir_base = os.path.join(img_dir_base, "invertY")

    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(img_dir_base, exist_ok=True)

    for grid_size in grid_sizes:
        print(f"Current Grid-size: {grid_size}")
        numpoints = grid_size[0] * grid_size[1] * 10

        # File and pathnames
        load_pathname = "G:\\ukr_data\\Einzelzaehne_sorted\\"

        max_exp_cfg_path = glob.glob(os.path.join(load_pathname, f"*{numpoints}*"))

        # Calculate the max expansion of all teeth for normalization
        if not max_exp_cfg_path:
            expansion_max = dp.calc_max_expansion(
                load_dir=load_pathname,
                z_threshold=z_threshold,
                numpoints=numpoints,
                save_dir=load_pathname,
                save_as=["json", "npz"]
            )
            print(expansion_max)
        else:
            if max_exp_cfg_path[0].split(".")[-1] == "npz":
                expansion_max = np.load(max_exp_cfg_path[0])["expansion_max"]
            elif max_exp_cfg_path[0].split(".")[-1] == "json":
                with open(max_exp_cfg_path[0]) as f:
                    expansion_max = np.array(json.load(f)["expansion_max"])


        param_hash = dp.create_param_sha256(frame_size=frame_size,
                                            expansion_max=expansion_max,
                                            nan_val=nan_val,
                                            z_threshold=z_threshold) 
        print(param_hash)
        grid_folder = f"{grid_size[0]}x{grid_size[1]}"
        save_pathname = os.path.join(pcd_dir, param_hash, grid_folder)
        save_path_pcd = os.path.join(save_pathname, "pcd_grid")

        img_dir = os.path.join(img_dir_base, param_hash, "grayscale",
                            grid_folder, "img")

        img_dir_rgb = img_dir.replace("grayscale", "rgb")

        np_savepath = os.path.join(
            save_pathname,
            f"einzelzaehne_train_lb{normbounds[0]}_ub{normbounds[1]}_{param_hash}.npy",
        )

        params = dp.search_pcd_cfg(param_hash=param_hash)

        # Load the unregular .pcd files and save them as regularized grid pcds
        if not os.path.exists(save_path_pcd):
            params = dp.create_params_cfg(frame_size=frame_size,
                                        expansion_max=expansion_max,
                                        nan_val=nan_val,
                                        z_threshold=z_threshold, save_as=["json", "npz"])

            files = glob.glob(os.path.join(load_pathname, "*.stl"))
            for filename, num in zip(
                    files,
                    tqdm(
                        range(len(files)),
                        desc="Creating pcd-grid files..",
                        ascii=False,
                        ncols=100,
                    ),
            ):
                dp.pcd_to_grid(
                    filepath_stl=filename,
                    save_path_pcd=os.path.join(
                        save_path_pcd, f"einzelzahn_grid_{num}_{param_hash}.pcd"),
                    grid_size=grid_size,
                    expansion_max=expansion_max,
                    frame_size=frame_size,
                    nan_val=nan_val,
                    plot_bool=False,
                    numpoints=numpoints,
                    z_threshold=z_threshold,
                    rotation_deg_xyz=rotation_deg_xyz,
                    invertY =  invertY
                )

        # Convert the 3D regularized grid pcds to one 2D numpy array for training
        if not os.path.exists(np_savepath):
            dp.grid_pcd_to_2D_np(
                pcd_dirname=save_path_pcd,
                np_savepath=np_savepath,
                grid_size=grid_size,
                z_threshold=z_threshold,
                nan_val=nan_val,
                normbounds=normbounds,
            )

        # Convert the 2D numpy array to grayscale images for nvidia stylegan
        if not os.path.exists(img_dir):
            # Creating grayscale images
            dp.np_grid_to_grayscale_png(npy_path=np_savepath,
                                        img_dir=img_dir,
                                        param_hash=param_hash)
        else:
            print(
                    f"L-PNGs for this parameter-set already exist at: {img_dir}"
                )

        if not os.path.exists(img_dir_rgb):
            # Converting to rgb and save in different folder
            dp.image_conversion_L_RGB(img_dir=img_dir, rgb_dir=img_dir_rgb)
        else:
            print(
            f"RGB-PNGs for this parameter-set already exist at: {img_dir_rgb}"
            )
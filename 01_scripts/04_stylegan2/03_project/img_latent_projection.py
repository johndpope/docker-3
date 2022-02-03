import glob
import os
from matplotlib.pyplot import grid
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from gan_tools.get_min_metric import *
import pcd_tools.data_processing as dp

t0 = time.time()

# For each results folder in p_results_dir: 
# if 1: Only use snapshot with minimal Error metric,
# if 0: Generate for all snapshots
metric_min_search_snapshot = 1

# if 1: Only use the folder and the matching snapshot with the global minimal error metric
# if 0: Use all folders in p_results_dir
metric_min_search_folder = 0
dry_run = 0

# Select kimg for current config, set to None if there is no kimg sub-folder in the results dir
kimg = 3000
# Grid size for current config
grid_size = 256

# Paths
stylegan_path = "/home/home_bra/01_scripts/modules/stylegan2_ada_bra"
project_function = "projector_bra.py"

p_path_base = "/home/proj_depo/docker/models/stylegan2"
p_folder = "220202_ffhq-res256-mirror-paper256-noaug"
p_latent_dir_base = "/home/proj_depo/docker/data/einzelzahn/latents"
p_img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"

# Get the param hash
with open(os.path.join(p_path_base, p_folder, "img_path.txt")) as f:
    img_dir = f.read().replace("img_prep", "img")

param_hash = dp.get_param_hash_from_img_path(img_dir=img_dir)
img_dirs = [img_dir for img_dir in glob.glob(os.path.join(p_img_dir_base, "*")) if param_hash in img_dir]

for img_dir in img_dirs:
    img_dir = os.path.join(img_dir, "rgb", f"{grid_size}x{grid_size}", "img")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    img_folder = img_dir.split("images/")[-1].split("/")[0]

    if kimg:
        p_results_dir = os.path.join(p_path_base, p_folder, "results", f"kimg{kimg:04d}")
    else:
        p_results_dir = os.path.join(p_path_base, p_folder, "results")

    if metric_min_search_folder:
        metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_dir, as_dataframe=True)
        results_folder_list = [ metric_list.iloc[0]["Folder"] ]
    else:
        # # Find the number after kimg in the results_folder string
        # # kimg = int([match.split("kimg")[-1] for match in results_folder.split("-") if "kimg" in match][0])
        # kimg = int(results_folder.split("kimg")[-1].split("-")[0])
        results_folder_list = os.listdir(p_results_dir)
        
    for results_folder in results_folder_list:

        p_run_dir = os.path.join(p_results_dir, results_folder)

        if metric_min_search_snapshot:
            snapshot_name, metric_min, _ = get_min_metric(p_run_dir)
            network_pkl_path_list = glob.glob(os.path.join(p_run_dir,
                                                    f"{snapshot_name}*"))
            print(f"\nMetric:     {metric_min}")
        else:
            network_pkl_path_list = sorted(glob.glob(os.path.join(p_run_dir, "*.pkl")))


        for network_pkl_path in network_pkl_path_list:

            print(f"    Folder:     {results_folder}")
            print(f"    Snapshot:   {os.path.basename(network_pkl_path).split('.')[0]}")
            print(f"    Images:     {img_folder}")
            
            if kimg:
                latent_dir = os.path.join(p_latent_dir_base, img_folder, f"{grid_size}x{grid_size}", p_folder, f"kimg{kimg:04d}", results_folder, os.path.basename(network_pkl_path).split(".")[0], "latent")  
            else:
                latent_dir = os.path.join(p_latent_dir_base, img_folder, f"{grid_size}x{grid_size}", p_folder, results_folder, os.path.basename(network_pkl_path).split(".")[0], "latent")  
            if not dry_run:
                for img_path in img_paths:   
                    latent_dir_img = os.path.join(latent_dir, os.path.basename(img_path).split(".")[0])
                    if not os.path.exists(latent_dir_img) or not os.listdir(latent_dir_img):
                        os.system(
                            f'python {os.path.join(stylegan_path, project_function)} \
                            --outdir={latent_dir_img} \
                            --target={img_path} \
                            --save-video=False \
                            --network={network_pkl_path}'
                            )
                    else:
                        print(f"    {os.path.basename(img_path)} already existing.\n")

    print(f"Elapsed time in seconds: {time.time()-t0}")
            
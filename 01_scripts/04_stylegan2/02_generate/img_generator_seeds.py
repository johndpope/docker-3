import glob
import os
from matplotlib.pyplot import grid
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from gan_tools.get_min_metric import *

t0 = time.time()

# Seed for image generation 
# if "num1-num2" then seeds = range(num1, num2+1)
seed = "0-999"

# For each results folder in p_results_dir: 
# if 1: Only use snapshot with minimal Error metric,
# if 0: Generate for all snapshots
metric_min_search_snapshot = 1

# if 1: Only use the folder and the matching snapshot with the global minimal error metric
# if 0: Use all folders in p_results_dir
metric_min_search_folder = 1
dry_run = 0

# Select kimg for current config, set to None if there is no kimg sub-folder in the results dir
kimg = 750
# Grid size for current config
grid_size = 256

# Paths
stylegan_path = "/home/home_bra/01_scripts/modules/stylegan2_ada_bra"
generate_function = "generate.py"
p_path_base = "/home/proj_depo/docker/models/stylegan2"
folder = "220211_ffhq-res256-mirror-paper256-noaug" #"220208_ffhq-res256-mirror-paper256-noaug"

if kimg:
    p_results_dir = os.path.join(p_path_base, folder, "results", f"kimg{kimg:04d}")
else:
    p_results_dir = os.path.join(p_path_base, folder, "results")

p_img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images/images-generated"

if type(seed).__name__ == "str":
    nums = seed.split("-")
    iterator_obj = range(int(nums[0]), int(nums[1])+1)


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
        print(f"    Seeds:      {seed} (second number included)")

        if kimg:
            img_dir = os.path.join(p_img_dir_base, f"{grid_size}x{grid_size}", folder, f"kimg{kimg:04d}", results_folder, os.path.basename(network_pkl_path).split(".")[0], "img")  
        else:
            img_dir = os.path.join(p_img_dir_base, f"{grid_size}x{grid_size}", folder, results_folder, os.path.basename(network_pkl_path).split(".")[0], "img")  

        if not dry_run and \
            (not os.path.exists(img_dir) \
            or len(glob.glob(os.path.join(img_dir, "*.png"))) < len(iterator_obj)):

            os.makedirs(img_dir, exist_ok=True)
            os.system(f'python {os.path.join(stylegan_path, generate_function)} \
                --outdir={img_dir} \
                --network={network_pkl_path} \
                --seeds={seed}')
        else:
            print("    Images already existing.\n")

print(f"Elapsed time in seconds: {time.time()-t0}")
        

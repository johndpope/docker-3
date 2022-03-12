import glob
import os
from matplotlib.pyplot import grid
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from gan_tools.get_min_metric import *
import dnnlib.util as util

t0 = time.time()

# --------------------------------------------------------------------------------------- #

stylegan_version = 1
stylegan_versions = ["stylegan2_ada", "stylegan2-ada-pytorch", "stylegan3",]

## User Input
dry_run = False


# Select kimg for current config, set to None if there is no kimg sub-folder in the results dir
kimg = 50000  # Set to None if p_results_dir = os.path.join(p_path_base, folder, "results")
default_folder = "220311_ffhq-res256-mirror-paper256-noaug" #"220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"

# Seed for image generation 
# if "num1-num2" then seeds = range(num1, num2+1)
seed = 100

# For each results folder in p_results_dir: 
# if True: Only use snapshot with minimal Error metric,
# if False: Generate for all snapshots
metric_min_search_snapshot = True

# if True: Only use the folder and the matching snapshot with the global minimal error metric
# if False: Use all folders in p_results_dir
metric_min_search_folder = False



# Grid size for current config --> Look up in img_path.txt
grid_size = 256

# --------------------------------------------------------------------------------------- #

## Paths
stylegan_path = f"/home/home_bra/01_scripts/modules/{stylegan_versions[stylegan_version]}_bra"
# stylegan_path = "/home/home_bra/01_scripts/modules/stylegan2_ada_bra"
generate_function = "generate.py"
p_path_base = "/home/proj_depo/docker/models/stylegan2"
p_img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images/images-generated"

# --------------------------------------------------------------------------------------- #

if dry_run:
    print("*-----------*")
    print("DRY RUN")
    print("*-----------*")
else:
    print("*-----------*")
    print("TRAINING RUN")
    print("*-----------*")

last_folder = os.path.basename(sorted(os.listdir(p_path_base))[-1])
kimg_str = f"kimg{kimg:04d}" if kimg is not None else None

if default_folder is not None: 
    print(f"Using default folder: {default_folder}")
    folder = default_folder
elif util.ask_yes_no(f"Use last-folder: {last_folder} "):
    folder = last_folder
else:
    folder = str(
        input(
            "Input folder-name to use: \n"
        ))

if not folder:
    raise ValueError("foldername is empty")

if kimg is not None:
    p_results_dir = os.path.join(p_path_base, folder, "results", kimg_str)
else:
    p_results_dir = os.path.join(p_path_base, folder, "results")


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
        snapshot_name, metric_min, _, _ = get_min_metric(p_run_dir)
        network_pkl_path_list = glob.glob(os.path.join(p_run_dir,
                                                f"{snapshot_name}*"))
        print(f"\nMetric:     {metric_min}")
    else:
        network_pkl_path_list = sorted(glob.glob(os.path.join(p_run_dir, "*.pkl")))


    for network_pkl_path in network_pkl_path_list:

        print(f"    Folder:     {results_folder}")
        print(f"    Snapshot:   {os.path.basename(network_pkl_path).split('.')[0]}")
        print(f"    Seeds:      {seed} (second number included)")

        if kimg is not None:
            img_dir = os.path.join(p_img_dir_base, f"{grid_size}x{grid_size}", folder, kimg_str, results_folder, os.path.basename(network_pkl_path).split(".")[0], "img")  
        else:
            img_dir = os.path.join(p_img_dir_base, f"{grid_size}x{grid_size}", folder, results_folder, os.path.basename(network_pkl_path).split(".")[0], "img")  

        if not dry_run:
            if not os.path.exists(img_dir) or not os.listdir(img_dir):

                os.makedirs(img_dir, exist_ok=True)
                os.system(f'python {os.path.join(stylegan_path, generate_function)} \
                    --outdir={img_dir} \
                    --network={network_pkl_path} \
                    --seeds={seed}')
            else:
                print("Images already existing.\n")

print(f"Elapsed time in seconds: {time.time()-t0}")

print(f"Image Directory:\n{img_dir}")

if dry_run:
    print("Dry run finished. No errors.")
else:
    print("Generate finished. No errors.")       

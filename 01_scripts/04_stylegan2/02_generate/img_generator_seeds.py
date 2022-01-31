import glob
import os
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from gan_tools.get_min_metric import *

print(sys.path)
metric_min_search_snapshot = 1
metric_min_search_folder = 1
dry_run = 0
kimg = 750

p_path_base = "/home/proj_depo/docker/models/stylegan2/"
folder = "220118_ffhq-res256-mirror-paper256-noaug"

# Seed for image generation 
# if "num1-num2" then seeds = range(num1, num2+1)
seed = "0-91"

# if type(seed).__name__ == "str":
#     nums = seed.split("-")
#     iterator_obj = range(int(nums[0]), int(nums[1])+1)

# Paths

stylegan_path = "/home/home_bra/01_scripts/modules/stylegan2_ada_bra"
generate_function = "generate.py"

p_results_dir = os.path.join(p_path_base, folder, "results", f"kimg{kimg:04d}")

if metric_min_search_folder:
    metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_dir, as_dataframe=True)
    results_folder = metric_list.iloc[0]["Folder"]
    print(results_folder)
else:
    results_folder = "00002-img_prep-mirror-paper256-kimg3000-ada-target0.5-bgcfnc-nocmethod-resumecustom-freezed0"
    # Find the number after kimg in the results_folder string
    # kimg = int([match.split("kimg")[-1] for match in results_folder.split("-") if "kimg" in match][0])
    kimg = int(results_folder.split("kimg")[-1].split("-")[0])

p_run_dir = os.path.join(p_path_base, folder, "results", f"kimg{kimg:04d}", results_folder)


if metric_min_search_snapshot:
    snapshot_name, metric_min, _ = get_min_metric(p_run_dir)
    network_pkl_path_list = glob.glob(os.path.join(p_run_dir,
                                              f"{snapshot_name}*"))
    print(f"Metric:     {metric_min}")
else:
    network_pkl_path_list = sorted(glob.glob(os.path.join(p_run_dir, "*.pkl")))


for network_pkl_path in network_pkl_path_list:
    print(f"Snapshot:   {os.path.basename(network_pkl_path).split('.')[0]}")
    print(f"Seeds:      {seed} (second number included)")

    if not dry_run:
        p_out_path = os.path.join(p_run_dir, "img_gen", os.path.basename(network_pkl_path).split(".")[0])
        os.makedirs(p_out_path, exist_ok=True)
        t1 = time.time()
        os.system(f'python {os.path.join(stylegan_path, generate_function)} \
            --outdir={p_out_path} \
            --network={network_pkl_path} \
            --seeds={seed}')

        print(f"Elapsed time in seconds: {time.time()-t1}")
    

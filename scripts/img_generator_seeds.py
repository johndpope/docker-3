import glob
import os
import numpy as np
import time

from get_min_metric import get_min_metric
os.system("clear")

metric_min_search = 1
dry_run = 1

# Seed for image generation 
# if "num1-num2" then seeds = range(num1, num2+1)
seed = 1

device = "gpu"

# if type(seed).__name__ == "str":
#     nums = seed.split("-")
#     iterator_obj = range(int(nums[0]), int(nums[1])+1)

# Paths
if device == "gpu":
    stylegan_path = "/home/home_bra/repo/stylegan2-ada-bugfixes"
    generate_function = "generate.py"
elif device == "cpu":
    stylegan_path = "/home/home_bra/repo/stylegan2-ada-cpu"
    generate_function = "generate_without_gpu.py"
else:
    raise ValueError("device must be in ['gpu', 'cpu']")

p_path_base = "/home/proj_depo/docker/models/stylegan2/"
folder = "220118_ffhq-res256-mirror-paper256-noaug"
results_folder = "00000-img_prep-mirror-paper256-kimg3000-ada-target0.5-bgc-nocmethod-resumecustom-freezed0"

# Find the number after kimg in the results_folder string
# kimg = int([match.split("kimg")[-1] for match in results_folder.split("-") if "kimg" in match][0])
kimg = int(results_folder.split("kimg")[-1].split("-")[0])

p_results_path = os.path.join(p_path_base, folder, "results", f"kimg{kimg:04d}",
                          results_folder)

if metric_min_search:
    snapshot_name, metric_min, _ = get_min_metric(p_results_path)
    network_pkl_path_list = glob.glob(os.path.join(p_results_path,
                                              f"{snapshot_name}*"))
    print(f"Metric:     {metric_min}")
else:
    network_pkl_path_list = sorted(glob.glob(os.path.join(p_results_path, "*.pkl")))


for network_pkl_path in network_pkl_path_list:
    print(f"Snapshot:   {os.path.basename(network_pkl_path).split('.')[0]}")
    print(f"Seeds:      {seed} (second number included)")

    if not dry_run:
        p_out_path = os.path.join(p_results_path, "img_gen", os.path.basename(network_pkl_path).split(".")[0])
        os.makedirs(p_out_path, exist_ok=True)
        t1 = time.time()
        os.system(f'python {os.path.join(stylegan_path, generate_function)} \
            --outdir={p_out_path} \
            --network={network_pkl_path} \
            --seeds={seed}')

        print(f"Elapsed time in seconds: {time.time()-t1}")
    

import glob
import os
import numpy as np
import time

from get_min_metric import get_min_metric

# Seed for image generation
seed = 10

device = "gpu"

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
folder = "211231_brecahad-mirror-paper512-ada"
results_folder = "00000-img_prep-stylegan2-kimg3000-ada-resumecustom-freezed0"

p_run_path = os.path.join(p_path_base, folder, "results", results_folder)

snapshot_name, metric_min = get_min_metric(p_run_path=p_run_path)

network_pkl_path = glob.glob(os.path.join(p_run_path, f"{snapshot_name}*"))[0]

print(f"Metric: {metric_min}")
print(f"Snapshot: {snapshot_name}")

p_out_path = os.path.join(p_run_path, "img_gen")

t1 = time.time()
# Generate the image if it doesn't already exist
if not os.path.exists(os.path.join(p_out_path, f'seed{seed:04d}.png')):
    os.system(f'python {os.path.join(stylegan_path, generate_function)} \
        --outdir={p_out_path} \
        --network={network_pkl_path} \
        --seeds={seed}')
else:
    print("Image already exists.")

print(f"Elapsed time in seconds: {time.time()-t1}")
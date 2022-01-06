import glob
import os
import numpy as np
import time

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

p_path = os.path.join(p_path_base, folder, "results", results_folder)
metric_file = glob.glob(os.path.join(p_path, "metric*.txt"))[0]

# Open file with metrics and save as var
with open(metric_file, "r") as f:
    textfile = f.readlines()

# Get all metrics
metrics = []
for line in range(len(textfile)):
    metrics.append(float(textfile[line].split("_full ")[-1].replace("\n", "")))

metrics = np.array(metrics)

# Calculate the (rolling) difference for the metric
diff_metrics = np.diff(metrics)

# Neglects snapshots after certain metric if it diverges (diff > threshold diff)
threshold_diff = 2
for ctr, diff_metric in enumerate(diff_metrics):
    diff_num = ctr
    if diff_metric > threshold_diff:
        print(diff_num)
        break

metrics = metrics[:diff_num + 2]

# Calculate the minimal metric in the converging list of metrics
metric_min = np.min(metrics)

# Get the index for the metric
snapshot_num = np.where(metrics == metric_min)[0][0]

# Select the matching snapshot
snapshot_name = textfile[snapshot_num].split("time")[0].replace(" ", "")
network_pkl_path = glob.glob(os.path.join(p_path, f"{snapshot_name}*"))[0]

print(f"Metric: {metric_min}")
print(f"Snapshot: {snapshot_name}")

p_out_path = os.path.join(p_path, "img_gen")

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
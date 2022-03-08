import glob
import os
from matplotlib.pyplot import grid
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from gan_tools.get_min_metric import *
import pcd_tools.data_processing as dp
import dnnlib.util as util


t0 = time.time()

# For each results folder in p_results_dir: 
# if 1: Only use snapshot with minimal Error metric,
# if 0: Generate for all snapshots
metric_min_search_snapshot = 0

# if 1: Only use the folder and the matching snapshot with the global minimal error metric
# if 0: Use all folders in p_results_dir
metric_min_search_folder = 0
dry_run = False
infinity_run = False
residuals_only = False

# Paths
stylegan_path = "/home/home_bra/01_scripts/modules/stylegan2_ada_bra"
project_function = "projector_bra.py"

p_latent_dir_base = "/home/proj_depo/docker/data/einzelzahn/latents"
p_img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
p_dir_base = "/home/proj_depo/docker/models/stylegan2/"

default_folder = None #"220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
last_folder = os.path.basename(sorted(os.listdir(p_dir_base))[-1])

# Select kimg for current config, set to None if all kimgs are needed
kimg_num = 10000

if not dry_run:
    # Set ENVARS for CPU:XLA
    os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
    os.environ["XLA_FLAGS"]="--xla_hlo_profile"


if util.ask_yes_no(f"Use last-folder: {last_folder} "):
    folder = last_folder
elif default_folder is not None: 
    if util.ask_yes_no(f"Use default-folder: {default_folder} "):
        folder = default_folder
else:
    folder = str(
        input(
            "Input folder-name to use: \n"
        ))
if not folder:
    raise ValueError("foldername is empty")

kimg = f"kimg{kimg_num:04d}" if kimg_num is not None else None

p_results_dir_base = os.path.join(p_dir_base, folder, "results")

if kimg is not None:
    p_results_dirs = [os.path.join(p_results_dir_base, kimg)]
else:
    p_results_dirs = [os.path.join(p_results_dir_base, kimg_folder) for kimg_folder in os.listdir(p_results_dir_base)] if os.listdir(p_results_dir_base)[0].startswith("kimg") else [p_results_dir_base]

# Get the param hash
with open(os.path.join(p_dir_base, folder, "img_path.txt")) as f:
    img_dir = f.read().replace("img_prep", "img")

param_hash = dp.get_param_hash_from_img_path(img_dir=img_dir)
img_dir_components = os.path.normpath(img_dir).split(os.sep)
grid_size = img_dir_components[-2]
img_folder = img_dir_components[-4]
if not "images-" in img_folder or not "x" in grid_size:
    raise ValueError("Check Image Folder Structure")

img_dirs = [img_dir]
# img_folders = [img_folder for img_folder in sorted(os.listdir(p_img_dir_base)) if img_folder in img_dir]
# img_dir = os.path.join(p_img_dir_base, img_folder, "rgb", grid_size, "img")
while 1:
    for p_results_dir in p_results_dirs:
        if metric_min_search_folder:
            metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_dir, as_dataframe=True)
            results_folder_list = [ metric_list.iloc[0]["Folder"] ]
        else:
            # # Find the number after kimg in the results_folder string
            # # kimg = int([match.split("kimg")[-1] for match in results_folder.split("-") if "kimg" in match][0])
            # kimg = int(results_folder.split("kimg")[-1].split("-")[0])
            results_folder_list = sorted(os.listdir(p_results_dir))
                
        for results_folder in results_folder_list:
            for img_dir in img_dirs: 
                img_resid_dir = img_dir+"_residual"
                # Add residual images to latent projection if reduced data set was used
                if os.path.exists(img_resid_dir):
                    img_paths=sorted(glob.glob(os.path.join(img_resid_dir, "*.png")))
                else:
                    img_paths = []

                if not residuals_only:
                    img_paths +=  sorted(glob.glob(os.path.join(img_dir, "*.png")))

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
                    print(f"    Images:     {img_folder}")
                    
                    
                    latent_dir = os.path.join(p_latent_dir_base, img_folder, grid_size, folder, results_folder, os.path.basename(network_pkl_path).split(".")[0], "latent")  

                    if not dry_run:
                        for img_path in img_paths:   
                            print(f"Image: {os.path.basename(img_path).split('.')[0]}")
                            latent_name = os.path.basename(img_path).split(".")[0]
                            if img_resid_dir in img_path:
                                latent_name += "_residual"

                            latent_dir_img = os.path.join(latent_dir, latent_name)

                            if not os.path.exists(latent_dir_img) or len(os.listdir(latent_dir_img))<4:
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
    if not infinity_run:
        break             
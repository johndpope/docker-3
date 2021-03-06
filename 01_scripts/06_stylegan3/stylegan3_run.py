import os
import glob
import datetime
import json
import sys
import argparse

from numpy import True_

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import dnnlib.util as util
from gan_tools.get_min_metric import get_min_metric_idx_from_dir, get_min_metric_list_from_dir

# ---------------------------------------------------------------------------------------------------------- #

## User Input

dry_run = False
resume_from_abort = False
parameter_study = True

# Run from cfg
# -------------- #
run_from_cfg = False
cfg_file_num = 0

run_from_cfg_list = False    # Set to False if 
cfg_file_metric_threshold = 0.013

cfg_file_folder = "220301_ffhq-res256-mirror-paper256-noaug"
cfg_file_kimg = 750
# -------------- #

# General train parameters
# -------------- #
grid = 256
img_folder = "images-4e742fa-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced89"

# Metric threshold for training resume after parameter study (kid) or run_from_cfg_list
metric_threshold = 0.02

# Fixed paths
stylegan_path = "/home/stylegan3"
home_path = "/home"
p_base_path = "/home/proj_depo/docker/models/stylegan3"
os.makedirs(p_base_path, exist_ok=True)
default_folder = None

## Parameters for training
# Iterables:

num_freezed_range = [0]
mirror_range = [True]
cfg_range = ["stylegan3-t"]
aug_range = ["ada"]
target_range = [0.5]
augpipe_range = ["bg"]
cmethod_range = ["bcr"]
metrics_range = [None]

# No iterables:
snap = 34
# Params
params = {}
params['kimg'] = 5000

params['metrics'] = None
params['pkl_url'] = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl"

# -------------- #

# ---------------------------------------------------------------------------------------------------------- #

if not (parameter_study or run_from_cfg):
    raise ValueError("Specify --parameter_study or --run_from_cfg.")
elif parameter_study and run_from_cfg:
    raise ValueError("parameter_study and run_from_cfg cannot both be True.")

if dry_run:
    print("*-----------*")
    print("DRY RUN")
    print("*-----------*")
else:
    print("*-----------*")
    print("TRAINING RUN")
    print("*-----------*")

# Define last folder
last_folder = sorted(os.listdir(p_base_path))[-1] if len(sorted(os.listdir(p_base_path)))>0 else None

if (run_from_cfg or run_from_cfg_list) and not parameter_study:
    cfg_file_kimg = f"kimg{cfg_file_kimg:04d}"

    if run_from_cfg_list:
        cfg_file_num_list = get_min_metric_idx_from_dir(p_results_dir=os.path.join(p_base_path, cfg_file_folder, "results", cfg_file_kimg), metric_threshold=cfg_file_metric_threshold)
    else:
        cfg_file_num_list = [cfg_file_num]

    print(f"cfg_file_num_list: {cfg_file_num_list}")

    cfg_file_dir = os.path.join(p_base_path, cfg_file_folder, "cfg", cfg_file_kimg)

    cfg_file_paths = [glob.glob(os.path.join(cfg_file_dir, f"{cfg_file_num:05d}_cfg*"))[0] for cfg_file_num in cfg_file_num_list]

    with open(cfg_file_paths[0]) as f:
        params = json.load(f)
        params.pop('img_path', None)

# Needed for folder selection/creation
if util.ask_yes_no("Create new folder <date>_<pkl-name>? "):
    today_var = datetime.date.today().strftime("%y%m%d")
    folder = f"{today_var}_{os.path.basename(params['pkl_url']).split('.')[-2]}"
    print(f"Foldername: {folder}")
elif last_folder is not None:
    if util.ask_yes_no(f"Use last-folder: {last_folder} "):
        folder = last_folder
elif util.ask_yes_no(f"Use default-folder: {default_folder} "):
    folder = default_folder
else:
    folder = str(
        input(
            "Input folder-name to use (Folder will be created if it doesnt exist!): \n"
        ))
if not folder:
    raise ValueError("foldername is empty")

p_path = os.path.join(p_base_path, folder)
p_results_base = os.path.join(p_path, "results")

if os.path.exists(p_path):
    print(f"Using existing folder: {folder}")

p_results = os.path.join(p_results_base, f"kimg{params['kimg']:04d}")
p_scripts = os.path.join(p_path, "scripts", f"kimg{params['kimg']:04d}")
p_cfg = os.path.join(p_path, "cfg", f"kimg{params['kimg']:04d}")

outdir = p_results

# Create directores
os.makedirs(p_path, exist_ok=True)

if not dry_run:
    os.makedirs(p_scripts, exist_ok=True)
    os.makedirs(p_results, exist_ok=True)
    os.makedirs(p_cfg, exist_ok=True)

# If pkl doesnt exist, download
if not glob.glob(os.path.join(p_path, "*.pkl")):
    # Download pickle for transfer learning
    os.system(f"wget -P {p_path} {params['pkl_url']}")

## Save pkl url to text
pkl_txt_name = "pkl_url.txt"
if not os.path.exists(os.path.join(p_path, pkl_txt_name)):
    with open(os.path.join(p_path, pkl_txt_name), "w") as f:
        f.write(params['pkl_url'])

## Load image path from file if it exists, else search for images
if not 'img_path' in params.keys() or params["img_path"] is None:
    img_path = f"/home/proj_depo/docker/data/einzelzahn/images/{img_folder}/rgb/{grid}x{grid}/img_prep.zip"
    params['img_path'] = img_path

img_txt_name = "img_path.txt"
if os.path.exists(os.path.join(p_path, img_txt_name)):
    print("Loading img_path from file..")
    with open(os.path.join(p_path, img_txt_name), "r") as f:
        params['img_path'] = f.readline()
else:
    # Create txt with img_path
    with open(os.path.join(p_path, img_txt_name), "w") as f:
        f.write(params['img_path'])

ctr = 0
idx_list = []
resumefile_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]
results_len = len(os.listdir(p_results)) if os.path.exists(p_results) else 0
kimg_len = len(os.listdir(p_results_base)) if os.path.exists(p_results_base) else 0
resume_from_loop_ctr = results_len

if results_len and resume_from_abort:
    num_folder = int(
        input("Input the folder number for training-resume: \n"))
    resumefile_path = sorted(
        glob.glob(
            os.path.join(
                glob.glob(os.path.join(p_results,
                                    f"{num_folder:05d}*"))[0],
                "*.pkl")))[-1]
    print(f"Resuming from {resumefile_path}")
elif results_len and not run_from_cfg:
    if util.ask_yes_no(f"Resume loop from ctr = {resume_from_loop_ctr}? "):
        print(f"Resuming from ctr = {resume_from_loop_ctr}")
elif not results_len and resume_from_abort:
    raise ValueError("Nothing to resume from.")
elif kimg_len>1:
    if util.ask_yes_no(f"Resume learning from metric_min models? "):
        # Get path of last "kimgxxxx" folder
        resume_dir = os.path.join(
            os.path.dirname(p_results),
            sorted(os.listdir(os.path.dirname(p_results)))[-2])
        # Get list with indices of the best configurations
        idx_list = get_min_metric_idx_from_dir(
            p_results_dir=resume_dir, metric_threshold=metric_threshold)
        while True:
            print(f"Resuming from idx_list: {idx_list}")
            if util.ask_yes_no(f"Skip indices? "):
                idx_list.remove(int(input("Skip Index: \n")))
            else:
                break

for metrics in metrics_range:

    if parameter_study and not run_from_cfg:
        for num_freezed in num_freezed_range:
            for cfg in cfg_range:
                for mirror in mirror_range:
                    for aug in aug_range:
                        for target in target_range:
                            for augpipe in augpipe_range:
                                for cmethod in cmethod_range:

                                    # Start from last folder if resume_from_loop_ctr or just rerun the indices of the best configs from the last run
                                    if ctr < resume_from_loop_ctr or (
                                            ctr not in idx_list and idx_list):
                                        ctr += 1
                                        continue

                                    params['aug'] = aug
                                    params['mirror'] = mirror
                                    params['cfg'] = cfg
                                    params['num_freezed'] = num_freezed
                                    params['target'] = target
                                    params['augpipe'] = augpipe
                                    params['cmethod'] = cmethod


                                    print("------")
                                    print(f"run: {ctr:02d}")
                                    print(f"kimg: {params['kimg']}")
                                    print(f"aug: {params['aug']}")
                                    print(f"mirror: {params['mirror']}")
                                    print(f"cfg: {params['cfg']}")
                                    print(f"num_Freezed: {params['num_freezed'] }")
                                    print(f"target: {params['target']}")
                                    print(f"augpipe: {params['augpipe']}")
                                    print(f"cmethod: {params['cmethod']}")
                                    print(f"metrics: {params['metrics']}")
                                    ctr += 1

                                    if not dry_run:

                                        save_ctr = len(
                                            glob.glob(os.path.join(p_results, '*')))
                                        # Save run-params as json cfg
                                        with open(
                                                os.path.join(
                                                    p_cfg, f"{save_ctr:05d}_cfg.json"),
                                                "w") as f:
                                            json.dump(params, f)

                                        # Copy this file to Model location
                                        scriptpath_file_p = os.path.join(
                                            p_scripts,
                                            f"{save_ctr:05d}_{os.path.basename(__file__)}"
                                        )
                                        os.system(f'cp {__file__} {scriptpath_file_p}')

                                        # Start training
                                        os.system(
                                            f"python {os.path.join(stylegan_path, 'train.py')} \
                                            --gpus=1 \
                                            --batch=32 \
                                            --gamma=2\
                                            --resume={resumefile_path} \
                                            --snap={snap}  \
                                            --data={params['img_path']} \
                                            --mirror={params['mirror']}\
                                            --kimg={params['kimg']} \
                                            --outdir={outdir} \
                                            --cfg={params['cfg']} \
                                            --aug={params['aug']} \
                                            --metrics={params['metrics']}")


    elif (run_from_cfg or run_from_cfg_list) and not parameter_study:
        for cfg_file_path in cfg_file_paths:
            img_path = params["img_path"]
            with open(cfg_file_path) as f:
                params = json.load(f)
                params.pop('img_path', None)
            params["img_path"] = img_path
            params["metrics"] = metrics

            print("------")
            print(f"kimg: {params['kimg']}")
            print(f"aug: {params['aug']}")
            print(f"mirror: {params['mirror']}")
            print(f"cfg: {params['cfg']}")
            print(f"num_Freezed: {params['num_freezed'] }")
            print(f"target: {params['target']}")
            print(f"augpipe: {params['augpipe']}")
            print(f"cmethod: {params['cmethod']}")
            print(f"metrics: {params['metrics']}")
            if not dry_run:
                save_ctr = len(
                    glob.glob(os.path.join(p_results, '*')))
                # Save run-params as json cfg
                with open(
                        os.path.join(
                            p_cfg, f"{save_ctr:05d}_cfg.json"),
                        "w") as f:
                    json.dump(params, f)

                # Copy this file to Model location
                scriptpath_file_p = os.path.join(
                    p_scripts,
                    f"{save_ctr:05d}_{os.path.basename(__file__)}"
                )
                os.system(f'cp {__file__} {scriptpath_file_p}')

                # Start training
                os.system(
                    f"python {os.path.join(stylegan_path, 'train.py')} \
                    --gpus=8 \
                    --resume={resumefile_path} \
                    --freezed={params['num_freezed']} \
                    --snap={snap}  \
                    --data={params['img_path']} \
                    --mirror={params['mirror']}\
                    --kimg={params['kimg']} \
                    --outdir={outdir} \
                    --cfg={params['cfg']} \
                    --aug={params['aug']} \
                    --target={params['target']} \
                    --augpipe={params['augpipe']} \
                    --cmethod={params['cmethod']} \
                    --metrics={params['metrics']}")
        

if dry_run:
    print("Dry run finished. No errors.")
else:
    print("Training finished. No errors.")

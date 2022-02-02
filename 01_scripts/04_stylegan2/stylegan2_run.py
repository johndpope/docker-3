import os
import glob
import datetime
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import dnnlib.util as util
from gan_tools.get_min_metric import get_min_metric_idx_from_dir

resume_from_abort = False
parameter_study = False
run_from_cfg = True
dry_run = False

if dry_run:
    print("*-----------*")
    print("DRY RUN")
    print("*-----------*")
else:
    print("*-----------*")
    print("TRAINING RUN")
    print("*-----------*")

grid = 256
img_folder = "56fa467-abs-keepRatioXY-invertY"

## Parameters for training
# Iterables:
num_freezed_range = [0, 1, 2]
mirror_range = [True]
cfg_range = ["paper256"]
aug_range = ["ada"]
target_range = [0.5, 0.6, 0.7]
augpipe_range = ["bgc", "bgcfnc"]
cmethod_range = ["nocmethod", "bcr"]

# No iterables:
snap = 34
# Params
params = {}
params['kimg'] = 3000
params['metrics'] = "kid50k_full"
params['pkl_url'] = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"

if run_from_cfg:
    cfg_file_num = 13
    cfg_file_dir = "/home/proj_depo/docker/models/stylegan2/220118_ffhq-res256-mirror-paper256-noaug/cfg/kimg3000"
    cfg_file_path = glob.glob(os.path.join(cfg_file_dir, f"*{cfg_file_num}*"))[0]
    with open(cfg_file_path) as f:
        params = json.load(f)
        params['img_path'] = None

# Metric threshold for training resume after parameter study (kid)
metric_threshold = 0.02

# Fixed paths
stylegan_path = "/home/stylegan2-ada"
home_path = "/home"
p_base_path = "/home/proj_depo/docker/models/stylegan2"
default_folder = None
last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])

# Needed for folder selection/creation
if util.ask_yes_no("Create new folder <date>_<pkl-name>? "):
    today_var = datetime.date.today().strftime("%y%m%d")
    folder = f"{today_var}_{os.path.basename(params['pkl_url']).split('.')[-2]}"
    print(f"Foldername: {folder}")
elif util.ask_yes_no(f"Use last-folder: {last_folder} "):
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

if os.path.exists(p_path):
    print(f"Using existing folder: {folder}")

p_results = os.path.join(p_path, "results", f"kimg{params['kimg']:04d}")
p_scripts = os.path.join(p_path, "scripts", f"kimg{params['kimg']:04d}")
p_cfg = os.path.join(p_path, "cfg", f"kimg{params['kimg']:04d}")

outdir = p_results

# Create directores
os.makedirs(p_path, exist_ok=True)
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
img_txt_name = "img_path.txt"
if not params['img_path']:
    if os.path.exists(os.path.join(p_path, img_txt_name)):
        print("Loading img_path from file..")
        with open(os.path.join(p_path, img_txt_name), "r") as f:
            params['img_path'] = f.readline()
    else:
        img_paths = [
            x[0] for x in os.walk(
                f"/home/proj_depo/docker/data/einzelzahn/images/{img_folder}/rgb/{grid}x{grid}/"
            ) if os.path.basename(x[0]) == "img_prep"
        ]

        # If more than one param set exists: ask
        if len(img_paths) > 1:
            for num, img in enumerate(img_paths):
                print(f"Index {num}: {os.path.dirname(img)}")
            params['img_path'] = img[int(input(f"Enter Index for preferred img-Files: "))]
        else:
            params['img_path'] = img_paths[0]
        # Create txt with img_path
        with open(os.path.join(p_path, img_txt_name), "w") as f:
            f.write(params['img_path'])

ctr = 0
idx_list = []
resumefile_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]
results_len = len(os.listdir(p_results))
resume_from_loop_ctr = results_len

if results_len:
    if resume_from_abort:
        num_folder = int(
            input("Input the folder number for training-resume: \n"))
        resumefile_path = sorted(
            glob.glob(
                os.path.join(
                    glob.glob(os.path.join(p_results,
                                           f"{num_folder:05d}*"))[0],
                    "*.pkl")))[-1]
        print(f"Resuming from {resumefile_path}")

    if util.ask_yes_no(f"Resume loop from ctr = {resume_from_loop_ctr}? "):
        print(f"Resuming from ctr = {resume_from_loop_ctr}")
    else:
        resume_from_loop_ctr = 0

if len(os.listdir(os.path.dirname(p_results)))>1:
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

if parameter_study:
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

                                # instantiate an empty dict
                                params = {}
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
elif run_from_cfg:
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

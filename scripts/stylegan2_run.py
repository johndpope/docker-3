import os
import glob
import datetime
import shutil
import dnnlib.util as util

from get_min_metric import get_min_metric_idx_from_dir

resume_from_abort = False
dry_run = False

pkl_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
grid = 256
param_hash = "7a9a625"

## Parameters for training
# Iterables:
num_freezed_range = [0, 1, 2]
mirror_range = [False, True]
cfg_range = ["paper256","auto"]
aug_range = ["ada", "noaug"]
# No iterables:
snap = 34
kimg = 3000

# Metric threshold for training resume after parameter study
metric_threshold = 30

# Fixed paths
stylegan_path = "/home/stylegan2-ada"
home_path = "/home"
p_base_path = "/home/proj_depo/docker/models/stylegan2"
default_folder = None
last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])

# Needed for folder selection/creation
if util.ask_yes_no("Create new folder <date>_<pkl-name>? "):
    today_var = datetime.date.today().strftime("%y%m%d")
    folder = f"{today_var}_{os.path.basename(pkl_url).split('.')[-2]}"
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

p_results = os.path.join(p_path, "results")
p_scripts = os.path.join(p_path, "scripts")

# Create directores
os.makedirs(p_path, exist_ok=True)
os.makedirs(p_scripts, exist_ok=True)
os.makedirs(p_results, exist_ok=True)

# If pkl doesnt exist, download
if not glob.glob(os.path.join(p_path, "*.pkl")):
    # Download pickle for transfer learning
    os.system(f"wget -P {p_path} {pkl_url}")

## Load image path from file if it exists, else search for images
img_txt_name = "img_path.txt"
if os.path.exists(os.path.join(p_path, img_txt_name)):
    print("Loading img_path from file..")
    with open(os.path.join(p_path, img_txt_name), "r") as f:
        img_path = f.readline()
else:
    img_paths = [
        x[0] for x in os.walk(
            f"/home/proj_depo/docker/data/einzelzahn/images/{param_hash}/rgb/{grid}x{grid}/"
        ) if os.path.basename(x[0]) == "img_prep"
    ]

    # If more than one param set exists: ask
    if len(img_paths) > 1:
        for num, img in enumerate(img_paths):
            print(f"Index {num}: {os.path.dirname(img)}")
        img_path = img[int(input(f"Enter Index for preferred img-Files: "))]
    else:
        img_path = img_paths[0]
    # Create txt with img_path
    with open(os.path.join(p_path, img_txt_name), "w") as f:
        f.write(img_path)

if resume_from_abort:
    num_folder = int(input("Input the folder number for training-resume: \n"))
    resumefile_path = sorted(
        glob.glob(
            os.path.join(
                glob.glob(os.path.join(p_results, f"{num_folder:05d}*"))[0],
                "*.pkl")))[-1]
    print(f"Resuming from {resumefile_path}")
else:
    resumefile_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]

ctr = 0

resume_from_loop_ctr = len(os.listdir(p_results))

if util.ask_yes_no(f"Resume loop from ctr = {resume_from_loop_ctr}? "):
    print(f"Resuming from ctr = {resume_from_loop_ctr}")
else:
    resume_from_loop_ctr = 0

if util.ask_yes_no(f"Resume learning from metric_min models? "):
    idx_list = get_min_metric_idx_from_dir(p_results_dir=p_results, metric_threshold=metric_threshold)
    print(f"Resuming from idx_list: {idx_list}")
else:
    idx_list = []

for num_freezed in num_freezed_range:

    for aug in aug_range:

        for mirror in mirror_range:

            for cfg in cfg_range:

                if ctr < resume_from_loop_ctr or (ctr not in idx_list and idx_list):
                    ctr += 1
                    continue

                print("------")
                print(f"run: {ctr:02d}")
                print(f"kimg: {kimg}")
                print(f"aug: {aug}")
                print(f"mirror: {mirror}")
                print(f"cfg: {cfg}")
                print(f"num_Freezed: {num_freezed}")
                ctr += 1


                if not dry_run:
                    # Copy this file to Model location
                    scriptpath_file_p = os.path.join(
                        p_scripts,
                        f"{len(glob.glob(os.path.join(p_results, '*'))):05d}_{os.path.basename(__file__)}"
                    )
                    os.system(f'cp {__file__} {scriptpath_file_p}')

                    # Start training
                    os.system(
                        f"python {os.path.join(stylegan_path, 'train.py')} \
                        --gpus=8 \
                        --resume={resumefile_path} \
                        --freezed={num_freezed} \
                        --snap={snap}  \
                        --data={img_path} \
                        --mirror={mirror}\
                        --kimg={kimg} \
                        --outdir={p_results} \
                        --cfg={cfg} \
                        --aug={aug}")

if dry_run:
    print("Dry run finished. No errors.")
else:
    print("Training finished. No errors.")

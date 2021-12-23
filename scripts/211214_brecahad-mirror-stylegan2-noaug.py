import os
import glob
import datetime
import shutil
import dnnlib.util as util

# resume_from_abort = True
# modelfolder = "00003-img_prep-stylegan2-kimg1000-noaug-resumecustom-freezed5"

pkl_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/paper-fig11a-small-datasets/brecahad-mirror-stylegan2-noaug.pkl"
grid = 512
param_hash = "7a9a625"
mdl_cfg = "stylegan2"

# Fixed paths
stylegan_path = "/home/stylegan2-ada"
home_path = "/home"
default_folder = "211214_brecahad-mirror-stylegan2-noaug"

# Needed for folder selection/creation
if util.ask_yes_no("Create new folder <date>_<pkl-name>? "):
    today_var = datetime.date.today().strftime("%y%m%d")
    folder = f"{today_var}_{os.path.basename(pkl_url).split('.')[-2]}"
    print(f"Foldername: {folder}")
elif util.ask_yes_no(f"Use default-folder: {default_folder} "):
    folder = default_folder
else:
    folder = str(
        input(
            "Input folder-name to use (Folder will be created if it doesnt exist!): \n"
        ))
    if not folder:
        raise ValueError("foldername is empty")

p_path = os.path.join("/home/proj_depo/docker/models/stylegan2", folder)

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

# if resume_from_abort:
#     resumefile_path = sorted(
#         glob.glob(
#             os.path.join(p_results,
#                          sorted(os.listdir(p_results))[-1], "*.pkl")))[-1]
#     print(f"Resuming from {resumefile_path}")
# else:

resumefile_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]

for num_freezed in range(6):
    print(f"Freezed: {num_freezed}")

    # Copy this file to Model location
    scriptpath_file_p = os.path.join(
        p_scripts,
        f"{len(glob.glob(os.path.join(p_results, '*'))):05d}_{os.path.basename(__file__)}"
    )
    os.system(f'cp {__file__} {scriptpath_file_p}')

    # Start training
    os.system(f"python {os.path.join(stylegan_path, 'train.py')} \
        --gpus=8 \
        --resume={resumefile_path} \
        --freezed={num_freezed} \
        --snap=20  \
        --data={img_path} \
        --mirror=False\
        --kimg=2000 \
        --outdir={p_results} \
        --cfg={mdl_cfg} \
        --aug=ada")

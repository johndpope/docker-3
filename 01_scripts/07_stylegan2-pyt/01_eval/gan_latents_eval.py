import pickle
import numpy as np
import glob
import os
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
import scipy
from hashlib import sha256
import time


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp
import dnnlib
import dnnlib.tflib as tflib
import gan_tools.gan_eval as gev


# Image parameters
grid_size = 256
p_path_base = "/home/proj_depo/docker/models/stylegan2"

# Number of images
real_file_num = 92
fake_file_num = 1000
rot_file_num = real_file_num   # = number of images per rotation

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Folders for the used network
p_folder = "220202_ffhq-res256-mirror-paper256-noaug"
kimg = "kimg3000"
results_cfg = "00000-img_prep-mirror-paper256-kimg3000-ada-target0.7-bgcfnc-nocmethod-resumecustom-freezed1"
snapshot = "network-snapshot-002924"

# Network hash (needed for caching the feature data)
fake_hash = sha256((p_folder+kimg+results_cfg+snapshot).encode()).hexdigest()[::10]

# Get the param hash
with open(os.path.join(p_path_base, p_folder, "img_path.txt")) as f:
    img_dir = f.read().replace("img_prep", "img")

param_hash = dp.get_param_hash_from_img_path(img_dir=img_dir)

# Paths
if os.name == "nt":
    img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{param_hash}\rgb\{grid_size}x{grid_size}\img"
    img_fake_dir_base = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\generated\{grid_size}x{grid_size}"
    img_rot_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"
elif os.name == "posix":
    img_lat_base = "/home/proj_depo/docker/data/einzelzahn/latents"
    img_rot_dir = img_lat_base
    # img_fake_dir_base = os.path.join(img_dir_base , "images-generated", f"{grid_size}x{grid_size}")

def get_lat_paths(img_lat_base, folder, grid_size, p_folder, kimg, results_cfg, snapshot):
    img_lat_dir = os.path.join(img_lat_base, folder, f"{grid_size}x{grid_size}", p_folder, kimg, results_cfg, snapshot, "latent")
    return [os.path.join(x[0], "dlatents.npz") for x in os.walk(img_lat_dir) if "img" in os.path.basename(x[0])]

def get_latents(img_lat_paths):
    img_lats = np.empty(shape=(0,512))
    for img_lat_path in img_lat_paths:
        img_lats = np.concatenate([img_lats, np.load(img_lat_path)["dlatents"][0,0,:].reshape(1,-1)], axis = 0)
    return img_lats

img_rot_folder = "images-56fa467-abs-keepRatioXY-invertY-rotated_x00_y00_z05"
img_real_folder = "images-56fa467-abs-keepRatioXY-invertY"

if not param_hash in img_rot_folder or not param_hash in img_real_folder:
    raise ValueError("Parameter Set from network doesn't match the latents.")


# Paths and labels for rot-img
# Get all paths from the "rotated" directories and append the items to list
img_rot_lat_paths = []
labels_rot = []
for rot_folder in sorted(os.listdir(img_rot_dir)):
    if "rotated" in rot_folder and param_hash in rot_folder:
        img_rot_lat_paths.extend(sorted(get_lat_paths(img_rot_dir, rot_folder, grid_size, p_folder, kimg, results_cfg, snapshot))[:rot_file_num])
        labels_rot.extend([rot_folder.split("rotated_")[-1]]*rot_file_num)

# Paths and labels for real-img
img_real_lat_paths = get_lat_paths(img_lat_base, img_real_folder, grid_size, p_folder, kimg, results_cfg, snapshot)[:real_file_num]
labels_real = ["real"] * len(img_real_lat_paths)

# Names for npy-data files
lat_real_path = os.path.join(data_dir, f"lats_real_{param_hash}_{real_file_num:04d}.npy")
# feat_fake_path = os.path.join(data_dir, f"feat_fake_{param_hash}_{fake_hash}_{fake_file_num:04d}.npy")
lat_rot_path = os.path.join(data_dir, f"lats_rot_{param_hash}_{rot_file_num:04d}_z{labels_rot[0].split('z')[-1]}-z{labels_rot[-1].split('z')[-1]}.npy")

# t0 = time.time()
if not os.path.exists(lat_real_path):
    img_real_lats = get_latents(img_real_lat_paths)
    np.save(lat_real_path, img_real_lats)
else:
    img_real_lats = np.load(lat_real_path)

if not os.path.exists(lat_rot_path):
    img_rot_lats = get_latents(img_rot_lat_paths)
    np.save(lat_rot_path, img_rot_lats)
else:
    img_rot_lats = np.load(lat_rot_path)

# print(time.time()-t0)
def expand_seed(seed, vector_size=512):
    rnd = np.random.RandomState(seed)
    return rnd.randn(1, vector_size)

if 1:
    # Show correlation structure
    print("Starting corr-print")
    img_num = 1

    features = np.empty(shape=(0,512))
    # features = expand_seed(img_num)

    features = np.concatenate([features, img_real_lats[img_num].reshape(1,-1)], axis=0)
    # for indx in range(img_num,len(img_rot_lats),real_file_num):
    #     features = np.concatenate([features, img_rot_lats[indx].reshape(1,-1)], axis=0)
    print(features.shape)
    # features = img_real_lats[0].reshape(1,-1)
 
    # Calculate correlation matrix
    corr_mat = np.corrcoef(features[:,:20], rowvar=False)

    # print(np.asarray(np.nonzero((corr_mat > 0.95)*(corr_mat < 0.999))))
    # print(corr_mat[(corr_mat > 0.95)*(corr_mat < 1)])

    # Plot correlation matrix
    plt.figure()
    plt.imshow(corr_mat)
    plt.colorbar()
    # Plot cluster map
    sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))
    plt.show()

# if 1:
#     gev.fit_tsne(   feat1 = img_real_lats, 
#                     feat2 = None, 
#                     label1 = labels_real, 
#                     label2 = None, 
#                     plt_bool=True)

# if 0:
#     gev.kdtree_query_ball_tree( feat1 = feat_real, 
#                                 feat2 = feat_fake, 
#                                 img_1_paths = img_real_paths, 
#                                 img_2_paths = img_fake_paths, 
#                                 label1 = labels_real,
#                                 label2 = labels_fake, 
#                                 max_distance = 3, 
#                                 plt_bool=False)




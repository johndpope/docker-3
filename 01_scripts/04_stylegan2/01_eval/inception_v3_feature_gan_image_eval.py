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
rot_file_num = real_file_num    # = number of images per rotation

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Folders for the used network
p_folder = "220208_ffhq-res256-mirror-paper256-noaug" #"220202_ffhq-res256-mirror-paper256-noaug"
kimg = "kimg3000"
results_cfg = "00000-img_prep-mirror-paper256-kimg3000-ada-target0.7-bgcfnc-nocmethod-resumecustom-freezed1"
snapshot = "network-snapshot-001392"

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
    img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
    img_real_dir = img_dir
    img_fake_dir_base = os.path.join(img_dir_base , "images-generated", f"{grid_size}x{grid_size}")
    img_rot_dir_base = img_dir_base

img_fake_dir = os.path.join(img_fake_dir_base, p_folder, kimg, results_cfg, snapshot, "img")

# Paths and labels for rot-img
# Get all paths from the "rotated" directories and append the items to list
if 0:
    img_rot_paths = []
    labels_rot = []
    for rot_folder in sorted(os.listdir(img_rot_dir_base)):
        if "rotated" in rot_folder and param_hash in rot_folder:
            img_rot_dir = os.path.join(img_rot_dir_base, rot_folder, "rgb", f"{grid_size}x{grid_size}", "img")
            img_rot_paths.extend(sorted(glob.glob(os.path.join(img_rot_dir, "*.png")))[:rot_file_num])
            labels_rot.extend([rot_folder.split("rotated_")[-1]]*rot_file_num)

# Paths and labels for real-img
img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:real_file_num]
labels_real = ["real"] * len(img_real_paths)

# Paths and labels for fake-img
img_fake_paths = sorted(glob.glob(os.path.join(img_fake_dir, "*.png")))[:fake_file_num]
labels_fake = ["fake"] * len(img_fake_paths)

# Names for npy-data files
feat_real_path = os.path.join(data_dir, f"feat_real_{param_hash}_{real_file_num:04d}.npy")
feat_fake_path = os.path.join(data_dir, f"feat_fake_{param_hash}_{fake_hash}_{fake_file_num:04d}.npy")
# feat_rot_path = os.path.join(data_dir, f"feat_rot_{param_hash}_{rot_file_num:04d}_z{labels_rot[0].split('z')[-1]}-z{labels_rot[-1].split('z')[-1]}.npy")

feature_net = None
# Init tf session and load feature if one of the required datasets doesnt yet exist
if not os.path.exists(feat_fake_path) or not os.path.exists(feat_real_path): # or not os.path.exists(feat_rot_path): 
    tflib.init_tf()
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)   

# Calculate the (x,2048) feature vectors
feat_real = gev.feature_net_calc(feat_path=feat_real_path, img_paths=img_real_paths, feature_net=feature_net)
feat_fake = gev.feature_net_calc(feat_path=feat_fake_path, img_paths=img_fake_paths, feature_net=feature_net)
# feat_rot = gev.feature_net_calc(feat_path=feat_rot_path, img_paths=img_rot_paths, feature_net=feature_net)

if 0:
    for z_angle, feat_rot_single in zip(range(5,46,5),np.array_split(feat_rot, np.ceil(feat_rot.shape[0]/real_file_num))):
        feat1=feat_real
        feat2=feat_rot_single
        print(f"\nZ-Rotation: {z_angle}")
        print(f"KID: {gev.compute_kid(feat1, feat2)}")
        print(f"FID: {gev.compute_fid(feat1, feat2)}")


if 0:
    # Show correlation structure
    print("Starting corr-print")
    features = np.concatenate([feat_real, feat_rot], axis=0)
    features = feat_real
    # Calculate correlation matrix
    corr_mat = np.corrcoef(features)

    # print(np.asarray(np.nonzero((corr_mat > 0.95)*(corr_mat < 0.999))))
    # print(corr_mat[(corr_mat > 0.95)*(corr_mat < 1)])

    # Plot correlation matrix
    plt.figure()
    plt.imshow(corr_mat)
    plt.colorbar()
    # Plot cluster map
    sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))
    plt.show()

if 0:
    gev.fit_tsne(   feat1 = feat_real, 
                    feat2 = feat_fake, 
                    label1 = labels_real, 
                    label2 = labels_fake, 
                    plt_bool=False)

if 1:
    gev.kdtree_query_ball_tree( feat1 = feat_real, 
                                feat2 = feat_fake, 
                                img_1_paths = img_real_paths, 
                                img_2_paths = img_fake_paths, 
                                label1 = labels_real ,
                                label2 = labels_fake, 
                                max_distance = 2, 
                                plt_bool=True)




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
import img_tools.image_processing as ip

# Image parameters
grid_size = 256
p_path_base = "/home/proj_depo/docker/models/stylegan2"

# Number of images
real_file_num = 92
fake_file_num = -1

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Folders for the used network
p_folder = "220224_ffhq-res256-mirror-paper256-noaug" #"220202_ffhq-res256-mirror-paper256-noaug"
kimg = 750
kimg_str = f"kimg{kimg:04d}"
results_cfg = "00001-img_prep-mirror-paper256-kimg750-ada-target0.5-bgc-nocmethod-resumecustom-freezed0"
snapshot = "network-snapshot-000418"

# Network hash (needed for caching the feature data)
fake_hash = sha256((p_folder+kimg_str+results_cfg+snapshot).encode()).hexdigest()[::10]

# Get the param hash
with open(os.path.join(p_path_base, p_folder, "img_path.txt")) as f:
    img_dir = f.read().replace("img_prep", "img")

param_hash = dp.get_param_hash_from_img_path(img_dir=img_dir)


# Paths
if os.name == "nt":
    img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{param_hash}\rgb\{grid_size}x{grid_size}\img"
    img_fake_dir_base = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\generated\{grid_size}x{grid_size}"
elif os.name == "posix":
    img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
    img_real_dir = img_dir
    img_fake_dir_base = os.path.join(img_dir_base , "images-generated", f"{grid_size}x{grid_size}")

img_fake_dir = os.path.join(img_fake_dir_base, p_folder, kimg_str, results_cfg, snapshot, "img_post")

if not os.path.exists(img_fake_dir):
    ip.ImagePostProcessing(img_dir=img_fake_dir.replace("img_post", "img"), img_new_dir = img_fake_dir)


# Paths and labels for real-img
img_real_paths = sorted(glob.glob(os.path.join(img_real_dir, "*.png")))
if real_file_num != -1:
    img_fake_paths = img_real_paths[:real_file_num]

labels_real = ["real"] * len(img_real_paths)

# Paths and labels for fake-img
img_fake_paths = sorted(glob.glob(os.path.join(img_fake_dir, "*.png")))
if fake_file_num != -1:
    img_fake_paths = img_fake_paths[:fake_file_num]

labels_fake = ["fake"] * len(img_fake_paths)

# Names for npy-data files
feat_real_path = os.path.join(data_dir, f"feat_real_{param_hash}_{len(img_real_paths):04d}.npy")
feat_fake_path = os.path.join(data_dir, f"feat_fake_{param_hash}_{fake_hash}_{len(img_fake_paths):04d}.npy")

feature_net = None
# Init tf session and load feature if one of the required datasets doesnt yet exist
if not os.path.exists(feat_fake_path) or not os.path.exists(feat_real_path): # or not os.path.exists(feat_rot_path): 
    tflib.init_tf()
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)   

# Calculate the (x,2048) feature vectors
feat_real = gev.feature_net_calc(feat_path=feat_real_path, img_paths=img_real_paths, feature_net=feature_net)
feat_fake = gev.feature_net_calc(feat_path=feat_fake_path, img_paths=img_fake_paths, feature_net=feature_net)

# if 0:
#     for z_angle, feat_rot_single in zip(range(5,46,5),np.array_split(feat_rot, np.ceil(feat_rot.shape[0]/real_file_num))):
#         feat1=feat_real
#         feat2=feat_rot_single
#         print(f"\nZ-Rotation: {z_angle}")
#         print(f"KID: {gev.compute_kid(feat1, feat2)}")
#         print(f"FID: {gev.compute_fid(feat1, feat2)}")


if 0:
    # Show correlation structure
    print("Starting corr-print")
    # features = np.concatenate([feat_real, feat_fake], axis=0)
    for features in [feat_real, feat_fake]:
        print(features.shape)
        # features = feat_real
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

if 1:
    gev.fit_tsne(   feat1 = feat_fake, 
                    feat2 = None, 
                    label1 = labels_fake, 
                    label2 = labels_fake, 
                    plt_bool=True,
                    fig_path=os.path.join(figure_dir, f"tsne_{fake_hash}.pickle"))

if 0:
    gev.kdtree_query_ball_tree( feat1 = feat_real, 
                                feat2 = feat_fake, 
                                img_1_paths = img_real_paths, 
                                img_2_paths = img_fake_paths, 
                                label1 = labels_real ,
                                label2 = labels_fake, 
                                max_distance = 1, 
                                plt_bool=True)




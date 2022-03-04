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

# Paths
p_latent_dir_base = "/home/proj_depo/docker/data/einzelzahn/latents"
p_img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
p_dir_base = "/home/proj_depo/docker/models/stylegan2/"

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00000-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


snapshot_dir = os.path.join(p_latent_dir_base, image_folder, grid, stylegan_folder, run_folder)

snapshots = sorted(os.listdir(snapshot_dir))
img_names = sorted(os.listdir(os.path.join(snapshot_dir, snapshots[0], "latent")))
img_names = [img_names[0]]

dlatents_arr = np.empty(shape=(0, 512))
img_proj_paths = []
img_target_paths = []
dist_loss_paths = []
for snapshot in snapshots:
    for img_name in img_names:
        img_proj_paths.append(os.path.join(snapshot_dir, snapshot, "latent", img_name, "proj.png"))
        img_target_paths.append(os.path.join(snapshot_dir, snapshot, "latent", img_name, "target.png"))
        dlatents_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dlatents.npz")
        dist_loss_paths.append(os.path.join(snapshot_dir, snapshot, "latent", img_name, "dist_loss.npz"))

        dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)

print(dlatents_arr.shape)
print(dist_loss_paths[0])
dist_losses = []
for dist_loss_path in dist_loss_paths:
    if os.path.exists(dist_loss_path):
        dist_losses.append(np.load(dist_loss_path))

for dist_loss in dist_losses:
    print(dist_loss["dist"], dist_loss["loss"])

# feature_net = None
# # Init tf session and load feature if one of the required datasets doesnt yet exist
# tflib.init_tf()
# with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
#     feature_net = pickle.load(f)   

# # Calculate the (x,2048) feature vectors
# feat_proj = gev.feature_net_calc(img_paths=img_proj_paths, feature_net=feature_net)
# feat_target = gev.feature_net_calc(img_paths=img_target_paths, feature_net=feature_net)

# if 1:
#     # Show correlation structure
#     print("Starting corr-print")

#     features = feat_target[0,:]

#     # Calculate correlation matrix
#     corr_mat = np.corrcoef(features)

#     # Plot correlation matrix
#     plt.figure()
#     plt.imshow(corr_mat)
#     plt.colorbar()
#     # Plot cluster map
#     sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))

#     features = feat_proj[0,:]

#     # Calculate correlation matrix
#     corr_mat = np.corrcoef(features)

#     plt.figure()
#     plt.imshow(corr_mat)
#     plt.colorbar()
#     # Plot cluster map
#     sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))


#     plt.show()


# gev.fit_tsne(   feat1 = feat_proj, 
#                 feat2 = None, 
#                 label1 = snapshots, 
#                 label2 = None, 
#                 plt_bool=True,
#                 fig_path=None,
#                 tsne_metric="euclidean")


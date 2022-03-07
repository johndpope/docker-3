import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
import sklearn.preprocessing as skp
import scipy
from hashlib import sha256

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp
from os_tools.import_paths import import_p_paths
import dnnlib
import dnnlib.tflib as tflib
import gan_tools.gan_eval as gev
import gan_tools.get_min_metric as gmm
import img_tools.image_processing as ip

exclude_snap0 = True
# Paths
p_style_dir_base, p_img_dir_base, p_latent_dir_base, p_cfg_dir = import_p_paths()


grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

p_run_dir = [x[0] for x in os.walk(os.path.join(p_style_dir_base, stylegan_folder)) if os.path.basename(x[0]) == run_folder][0]

metric_types = [metric_file.split("metric-")[-1].split(".")[0] for metric_file in os.listdir(p_run_dir) if "metric" in metric_file]
metrics_dict = {}

for metric_type in metric_types:
    _, _, _, metrics = gmm.get_min_metric(p_run_dir=p_run_dir, metric_type=metric_type)
    metrics_dict[metric_type] = np.array(metrics[1:] if exclude_snap0 else metrics)

snapshot_dir = os.path.join(p_latent_dir_base, image_folder, grid, stylegan_folder, run_folder)

snapshots = sorted(os.listdir(snapshot_dir))
if exclude_snap0:
    snapshots = snapshots[1:] # Exclude snapshot 0

img_names = sorted(os.listdir(os.path.join(snapshot_dir, snapshots[0], "latent")))
# img_names = [img_names[-2]]

snapshot_kimg = np.array([int(snapshot.split("-")[-1]) for snapshot in snapshots])[:, np.newaxis]

dlatents_arr = np.empty(shape=(0, 512))
img_proj_paths = []
img_target_paths = []
dist_loss_paths = []
dist_losses = []

latent_data = {}

for img_name in img_names:
    latent_data[img_name] = {}
    latent_data[img_name]["dist"] = np.empty(shape=(0,))
    latent_data[img_name]["loss"] = np.empty(shape=(0,))
    latent_data[img_name]["snapshots"] = snapshot_kimg

    for snapshot in snapshots:
        latent_data[img_name][snapshot] = {}
        img_proj_path=os.path.join(snapshot_dir, snapshot, "latent", img_name, "proj.png")
        img_target_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "target.png")
        dlatents_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dlatents.npz")
        dist_loss_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dist_loss.npz")
        latent_data[img_name]["dist"] = np.append(latent_data[img_name]["dist"], np.load(dist_loss_path)["dist"][0])
        latent_data[img_name]["loss"] = np.append(latent_data[img_name]["loss"], np.load(dist_loss_path)["loss"])
        dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)
        latent_data[img_name][snapshot] = {"img_proj_path": img_proj_path, "img_target_path": img_target_path, "dlatents_path": dlatents_path, "dist_loss_path": dist_loss_path}

plt.figure()
dist_list = []
# plt.title(name)
for name, item in latent_data.items():
    dist_list.append(item["dist"])
#     plt.plot(item["snapshots"], item["dist"])
# plt.show()
dist_arr = np.array(dist_list)
dist_mean = np.mean(dist_arr, axis=0)



# # Normalize with sklearn
# plt.plot(snapshot_kimg, skp.normalize(dist_mean[:, np.newaxis], norm="max",axis=0))
# plt.plot(snapshot_kimg, skp.normalize(metrics_dict["kid50k_full"][:, np.newaxis], norm="max", axis=0))
# Normalize with own fun
def normalize_01(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))
plt.plot(snapshot_kimg, normalize_01(dist_mean[:, np.newaxis]))
plt.plot(snapshot_kimg, normalize_01(metrics_dict["kid50k_full"][:, np.newaxis]))
plt.legend(["projector_loss-mean","kid50k_full"])

plt.show()



# print(dlatents_arr.shape)
# print(dist_loss_paths[0])
# dist_losses = []
# for dist_loss_path in dist_loss_paths:
#     if os.path.exists(dist_loss_path):
#         dist_losses.append(np.load(dist_loss_path))

# for dist_loss in dist_losses:
#     print(dist_loss["dist"], dist_loss["loss"])

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


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

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

import dnnlib
import dnnlib.tflib as tflib

# Switches for the script
corr_bool = 0 # create corrmat
tsne_bool = 0 # create tsne dimension-reduction
plt_bool = 0 # plot results
error_calc_bool = 0 # calculate kid error metric

# Image parameters
grid_size = 256
param_hash = "7a9a625"

# Number of images
real_file_num = 92
fake_file_num = 1000
rot_file_num = real_file_num    # = number of images per rotation

path_type = "docker" # from ["win", "docker"]

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Paths
if path_type == "win":
    img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\{param_hash}\rgb\{grid_size}x{grid_size}\img"
    img_fake_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\gen"
    img_rot_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\rotated"
elif path_type == "docker":
    img_real_dir = f"/home/proj_depo/docker/data/einzelzahn/images/{param_hash}/rgb/{grid_size}x{grid_size}/img"
    img_fake_dir = f"/home/proj_depo/docker/data/einzelzahn/images/generated/{grid_size}x{grid_size}/220118_ffhq-res256-mirror-paper256-noaug/kimg0750/00016-img_prep-mirror-paper256-kimg750-ada-target0.6-bgc-nocmethod-resumecustom-freezed1/network-snapshot-000278/img"
    img_rot_dir = "/home/proj_depo/docker/data/einzelzahn/images/rotated"
else:
    raise ValueError("Specify right path_type")

# Paths and labels for rot-img
# Get all paths from the "rotated" directories and append the items to list
img_rot_paths = []
labels_rot = []
for rot_folder in sorted(os.listdir(img_rot_dir)):
    img_rot_dir = os.path.join(img_rot_dir, rot_folder, param_hash,  "rgb", f"{grid_size}x{grid_size}", "img")
    img_rot_paths.extend(sorted(glob.glob(os.path.join(img_rot_dir, "*.png")))[:rot_file_num])
    labels_rot.extend([rot_folder]*rot_file_num)

# Paths and labels for real-img
img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:real_file_num]
labels_real = ["real"] * len(img_real_paths)

# Paths and labels for fake-img
img_fake_paths = sorted(glob.glob(os.path.join(img_fake_dir, "*.png")))[:fake_file_num]
labels_fake = ["fake"] * len(img_fake_paths)

# Names for npy-data files
feat_real_path = os.path.join(data_dir, f"feat_real_{real_file_num:04d}.npy")
feat_fake_path = os.path.join(data_dir, f"feat_fake_{fake_file_num:04d}.npy")
feat_rot_path = os.path.join(data_dir, f"feat_rot_{rot_file_num:04d}_z{labels_rot[0].split('z')[-1]}-z{labels_rot[-1].split('z')[-1]}.npy")

feature_net = None
# Init tf session and load feature if one of the required datasets doesnt yet exist
if not os.path.exists(feat_fake_path) or not os.path.exists(feat_real_path) or not os.path.exists(feat_rot_path): 
    tflib.init_tf()
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)   

def feature_net_calc(feat_path, img_paths, feature_net):
    # Load existing datasets or create features for reals, fakes and rots
    if os.path.exists(feat_path):   
        print(f"Loading from {feat_path} ..")
        features = np.load(feat_path)
    else:
        imgs = np.asarray(PIL.Image.open(img_paths[0]))[np.newaxis, :]
        for img_path in img_paths[1:]:
            img = np.asarray(PIL.Image.open(img_path))[np.newaxis, :]
            imgs = np.concatenate([imgs, img], axis=0)
        # Transpose for feature net
        imgs = imgs.transpose((0, 3, 1, 2))
        # Create batches
        features = np.empty(shape=(0,2048))
        for img_batch in np.array_split(imgs, np.ceil(imgs.shape[0]/100)):
            features = np.concatenate([features, feature_net.run(img_batch, assume_frozen=True)], axis=0)  
        # Save the features as npy
        np.save(feat_path, features)
    return features

# Calculate the (x,2048) feature vectors
feat_real = feature_net_calc(feat_path=feat_real_path, img_paths=img_real_paths, feature_net=feature_net)
feat_fake = feature_net_calc(feat_path=feat_fake_path, img_paths=img_fake_paths, feature_net=feature_net)
feat_rot = feature_net_calc(feat_path=feat_rot_path, img_paths=img_rot_paths, feature_net=feature_net)

def compute_kid(feat_real, feat_fake, num_subsets=1000, max_subset_size=1000):
    n = feat_real.shape[1]
    m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
        y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m

def compute_fid(feat_real, feat_fake):
    mu_fake = np.mean(feat_fake, axis=0)
    sigma_fake = np.cov(feat_fake, rowvar=False)
    mu_real = np.mean(feat_real, axis=0)
    sigma_real= np.cov(feat_real, rowvar=False)

    # Calculate FID.
    m = np.square(mu_fake - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
    dist = m + np.trace(sigma_fake + sigma_real - 2*s)

    return np.real(dist)

if error_calc_bool:
    for z_angle, feat_rot_single in zip(range(5,46,5),np.array_split(feat_rot, np.ceil(feat_rot.shape[0]/real_file_num))):
        feat1=feat_real
        feat2=feat_rot_single
        print(f"\nZ-Rotation: {z_angle}")
        print(f"KID: {compute_kid(feat1, feat2)}")
        print(f"FID: {compute_fid(feat1, feat2)}")

if corr_bool:
    # Show correlation structure
    print("Starting corr-print")
    features = np.concatenate([feat_real, feat_rot], axis=0)
    features = feat_real
    # Calculate correlation matrix
    corr_mat = np.corrcoef(features)

    # print(np.asarray(np.nonzero((corr_mat > 0.95)*(corr_mat < 0.999))))
    # print(corr_mat[(corr_mat > 0.95)*(corr_mat < 1)])

    # Plot correlation matrix
    if plt_bool:
        plt.figure()
        plt.imshow(corr_mat)
        plt.colorbar()
        # Plot cluster map
        sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))
        plt.show()

if tsne_bool:
    print("Starting tsne")

    images = np.concatenate([feat_real, feat_fake], axis=0)
    print(images[0,0].dtype)
    labels = labels_real + labels_fake
                    # learning_rate="auto",
    # Fit t-SNE to data
    tsne = TSNE(n_components=2,
                perplexity=10,
                metric="euclidean",
                method="exact",
                random_state=0,
                init='random')

    images_embedded = tsne.fit_transform(images)

    # Compare shape
    print(images.shape)
    print(images_embedded.shape)

    if plt_bool:
        # plt.imshow(images[0, :].reshape((grid_size, grid_size)))
        # Plot low-dimensional data
        plt.figure()
        sns.scatterplot(x=images_embedded[:, 0],
                        y=images_embedded[:, 1],
                        hue=labels,
                        legend='full',
                        palette='colorblind')

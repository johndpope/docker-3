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

# Switches for the script
corr_bool = 1 # create corrmat
tsne_bool = 1 # create tsne dimension-reduction
plt_bool = 1 # plot results
error_calc_bool = 0 # calculate kid error metric

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

p_folder = "220202_ffhq-res256-mirror-paper256-noaug"
kimg = "kimg3000"
results_cfg = "00000-img_prep-mirror-paper256-kimg3000-ada-target0.7-bgcfnc-nocmethod-resumecustom-freezed1"
snapshot = "network-snapshot-002924"

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
feat_rot_path = os.path.join(data_dir, f"feat_rot_{param_hash}_{rot_file_num:04d}_z{labels_rot[0].split('z')[-1]}-z{labels_rot[-1].split('z')[-1]}.npy")

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

    feat1 = feat_real
    label1 = labels_real
    img_1_paths = img_real_paths

    # feat2 = feat_real
    # label2 = labels_real
    # img_2_paths = img_real_paths

    feat2 = feat_fake
    label2 = labels_fake
    img_2_paths = img_fake_paths

    # feat2 = feat_rot
    # label2 = labels_rot
    # img_2_paths = img_rot_paths

    images = np.concatenate([feat1, feat2], axis=0)
    print(images[0,0].dtype)
    labels = label1 + label2
                    # learning_rate="auto",
    # Fit t-SNE to data
    tsne = TSNE(n_components=2,
                perplexity=10,
                metric="euclidean",
                method="exact",
                random_state=0,
                init='random')

    images_embedded = tsne.fit_transform(images)

    kdtree1 = scipy.spatial.KDTree(images_embedded[:feat1.shape[0]])
    kdtree2 = scipy.spatial.KDTree(images_embedded[feat1.shape[0]:])

    

    # neighbours = kdtree1.query_ball_tree(kdtree2, r=5)

    # feat1_feat2_pairs = []

    # for ctr, neighbour in enumerate(neighbours):
    #     if neighbour:
    #         for neighbour_sub in neighbour:    
    #             feat1_feat2_pairs.append([ctr, neighbour_sub])
    # print(len(feat1_feat2_pairs))
    # for feat1_feat2_pair in feat1_feat2_pairs[:200]:
    #     img1 = PIL.Image.open(img_1_paths[feat1_feat2_pair[0]])
    #     img2 = PIL.Image.open(img_2_paths[feat1_feat2_pair[1]])

    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     print([label1[feat1_feat2_pair[0]], label2[feat1_feat2_pair[1]]])
    #     print([os.path.basename(
    #             img_1_paths[feat1_feat2_pair[0]]
    #             ).split(".")[0],
    #         os.path.basename( 
    #             img_2_paths[feat1_feat2_pair[1]]
    #         ).split(".")[0]])

    #     ax1.imshow(img1)
    #     ax2.imshow(img2)
    #     plt.show()


    neigbours_self_2 = kdtree2.query_pairs(r=2)
    print(len(neigbours_self_2))

    for neigbour_self_2 in neigbours_self_2:
        img1 = PIL.Image.open(img_2_paths[neigbour_self_2[0]])
        img2 = PIL.Image.open(img_2_paths[neigbour_self_2[1]])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        print([label2[neigbour_self_2[0]], label2[neigbour_self_2[1]]])
        print([os.path.basename(
                img_2_paths[neigbour_self_2[0]]
                ).split(".")[0],
            os.path.basename( 
                img_2_paths[neigbour_self_2[1]]
            ).split(".")[0]])

        ax1.imshow(img1)
        ax2.imshow(img2)
        plt.show()

    # # Compare shape
    # print(images.shape)
    # print(images_embedded.shape)

    if plt_bool:
        # plt.imshow(images[0, :].reshape((grid_size, grid_size)))
        # Plot low-dimensional data
        plt.figure()
        sns.scatterplot(x=images_embedded[:, 0],
                        y=images_embedded[:, 1],
                        hue=labels,
                        legend='full',
                        palette='colorblind')

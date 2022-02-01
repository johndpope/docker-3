import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import PIL
import glob
from sklearn.datasets import load_digits
import seaborn as sns
import pandas as pd
import cv2
import plotly.graph_objects as go
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip
# import pcd_tools.data_processing as dp

tsne_bool = 1
corr_bool = 0
plt_bool = 0
gen_include = 1
rot_include = 0

grid_size = 256
file_num = 92

np_filepath = os.path.join(os.path.dirname(__file__), "mean_mat.npy")

rot_folders = sorted(
    os.listdir(
        r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\rotated"
    ))

img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
img_gen_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\models\stylegan2\220118_ffhq-res256-mirror-paper256-noaug\results\kimg3000\00002-img_prep-mirror-paper256-kimg3000-ada-target0.5-bgcfnc-nocmethod-resumecustom-freezed0\img_gen\network-snapshot-003000"


img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:file_num]


# real_labels = list(np.arange(len(img_real_paths)))

# img_real_paths = glob.glob(os.path.join(os.path.dirname(__file__), "real_same", "*.png"))[:file_num]
# img_real_paths =  glob.glob(os.path.join(os.path.dirname(__file__), "gen_same", "*.png"))[:file_num]

img_real_paths = random.sample(population=img_real_paths, k=file_num)

real_labels = ["real"] * len(img_real_paths)
# real_rot_labels = list(np.arange(len(img_real_paths)))

img_gen_paths = sorted(glob.glob(os.path.join(img_gen_dir, "*.png")))
img_gen_paths = random.sample(population=img_gen_paths, k=file_num)
gen_labels = ["gen"] * len(img_gen_paths)

if gen_include:
    img_paths = img_gen_paths# + img_real_paths
    labels = gen_labels# + real_labels
else:
    img_paths = img_real_paths
    labels = real_labels

if rot_include:
    for rot_folder in rot_folders:

        img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated\{rot_folders[0]}\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
        img_real_rot_paths = glob.glob(os.path.join(img_real_rot_dir,
                                                    "*.png"))[:file_num]
        img_real_rot_paths = random.sample(population=img_real_rot_paths,
                                           k=file_num)
        real_rot_labels = [f"{rot_folder}"] * len(img_real_rot_paths)
        img_paths += img_real_rot_paths
        labels += real_rot_labels


images = np.asarray(PIL.Image.open(img_paths[0]).convert("L")).T.reshape(1,-1)

for img_path in img_paths[1:]:
    images = np.concatenate([
        images,
        np.asarray(PIL.Image.open(img_path).convert("L")).T.reshape(1,-1)
    ],
                            axis=0)

# images = np.asarray(PIL.Image.open(img_paths[0])).T.reshape(3,-1).max(axis=0)[np.newaxis, :]

# for img_path in img_paths[1:]:
#     images = np.concatenate([
#         images,
#         np.asarray(PIL.Image.open(img_path)).T.reshape(3,-1).max(axis=0)[np.newaxis, :]
#     ],
#                             axis=0)

if tsne_bool:
    print("Starting tsne")

    # Fit t-SNE to data
    tsne = TSNE(n_components=2,
                perplexity=10,
                learning_rate="auto",
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
        if images_embedded.shape[1] == 2:
            plt.figure()
            sns.scatterplot(x=images_embedded[:, 0],
                            y=images_embedded[:, 1],
                            hue=labels,
                            legend='full',
                            palette='colorblind')
        elif images_embedded.shape[1] == 3:
            fig3d = go.Figure(
                go.Scatter3d(x=images_embedded[:, 0],
                             y=images_embedded[:, 1],
                             z=images_embedded[:, 2],
                             mode='markers',
                             marker=dict(size=3)))
            fig3d.show()

if corr_bool:
    # Show correlation structure
    print("Starting corr-print")
    # Calculate correlation matrix
    corr_mat = np.corrcoef(images)

    if plt_bool:
        # Plot correlation matrix
        plt.figure()
        plt.imshow(corr_mat)
        plt.colorbar()
        # Plot cluster map
        sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))

if plt_bool:
    plt.show()

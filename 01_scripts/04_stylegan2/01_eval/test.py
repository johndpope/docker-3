import numpy as np
from sklearn.manifold import TSNE
import sys
import os
import matplotlib.pyplot as plt
import PIL
import glob
from sklearn.datasets import load_digits
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.abspath("modules"))
import img_tools.image_processing as ip
# import pcd_tools.data_processing as dp

tsne_bool = 1
corr_bool = 0
plt_bool = 1

grid_size = 256
file_num = 92

np_filepath = os.path.join(os.path.dirname(__file__), "mean_mat.npy")

rot_folders = sorted(
    os.listdir(
        r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated"
    ))

img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
img_gen_dir = r"C:\Users\bra45451\Desktop\bra45451_offline\03_Design\01_Python\01_Scripts\0_Archiv\05_streamlit\images"

img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:file_num]
real_labels = ["real"] * len(img_real_paths)
# real_labels = list(np.arange(len(img_real_paths)))

# real_rot_labels = list(np.arange(len(img_real_paths)))

img_gen_paths = sorted(glob.glob(os.path.join(img_gen_dir, "*.png")))[:file_num]
gen_labels = ["gen"] * len(img_gen_paths)

img_paths = img_real_paths + img_gen_paths
labels = real_labels + gen_labels

# for rot_folder in rot_folders:
#     img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated\{rot_folders[0]}\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
#     img_real_rot_paths = glob.glob(os.path.join(img_real_rot_dir,
#                                                 "*.png"))[:file_num]
#     real_rot_labels = [f"{rot_folder}"] * len(img_real_rot_paths)
#     img_paths += img_real_rot_paths
#     labels += real_rot_labels

images = np.asarray(PIL.Image.open(img_paths[0]).convert("L")).reshape(1, -1)

for img_path in img_paths[1:]:
    images = np.concatenate([
        images,
        np.asarray(PIL.Image.open(img_path).convert("L")).reshape(1, -1)
    ],
                            axis=0)

print(images.shape)
img_reals = images[:92]
img_fakes = images[92:]
print(img_fakes.shape)
print(ip.compute_kid(feat_real =img_reals[:2], feat_fake = img_reals[:2]))

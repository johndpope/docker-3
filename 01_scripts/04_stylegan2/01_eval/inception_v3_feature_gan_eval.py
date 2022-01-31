import pickle
import numpy as np
import glob
import os
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

import dnnlib.tflib as tflib
import dnnlib

grid_size = 256

real_file_num = 92
fake_file_num = 20

path_type = "docker"

np_filepath = os.path.join(os.path.dirname(__file__), "mean_mat.npy")

if path_type == "win":
    img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
    img_gen_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\gen"
elif path_type == "docker":
    img_real_dir = f"/home/proj_depo/docker/data/einzelzahn/images/7a9a625/rgb/{grid_size}x{grid_size}/img"
    img_gen_dir = "/home/proj_depo/docker/data/einzelzahn/images/gen"
    img_gen_dir = "/home/proj_depo/docker/models/stylegan2/220118_ffhq-res256-mirror-paper256-noaug/results/kimg0750/00016-img_prep-mirror-paper256-kimg750-ada-target0.6-bgc-nocmethod-resumecustom-freezed1/img_gen/network-snapshot-000278"
else:
    raise ValueError("Specify right path_type")


img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:real_file_num]
real_labels = ["real"] * len(img_real_paths)
img_gen_paths = sorted(glob.glob(os.path.join(img_gen_dir, "*.png")))[:fake_file_num]
gen_labels = ["gen"] * len(img_gen_paths)

img_paths = img_real_paths + img_gen_paths


images = np.asarray(PIL.Image.open(img_paths[0]))[np.newaxis, :]

for img_path in img_paths[1:]:
    img = np.asarray(PIL.Image.open(img_path))[np.newaxis, :]
    images = np.concatenate([images, img], axis=0)

images = images.transpose((0, 3, 1, 2))

img_reals = images[:real_file_num]
img_fakes = images[real_file_num:]


tflib.init_tf()

with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    feature_net = pickle.load(f)



# num_gpus = 1
# max_reals = 10

feat_real = feature_net.run(img_reals, assume_frozen=True)
feat_fake = feature_net.run(img_fakes, assume_frozen=True)



print(feat_real.shape)
print(feat_fake.shape)
images = np.concatenate([feat_real, feat_fake], axis=0)

# Show correlation structure
print("Starting corr-print")
# Calculate correlation matrix
corr_mat = np.corrcoef(feat_fake)

print(np.asarray(np.nonzero((corr_mat > 0.95)*(corr_mat < 0.999))))
print(corr_mat[(corr_mat > 0.95)*(corr_mat < 1)])

# # Plot correlation matrix
# plt.figure()
# plt.imshow(corr_mat)
# plt.colorbar()
# # Plot cluster map
# sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))
# plt.show()
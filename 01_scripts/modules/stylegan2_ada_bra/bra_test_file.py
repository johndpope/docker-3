import dnnlib
import pickle
import numpy as np
import dnnlib.tflib as tflib
import glob
import os
import PIL
import matplotlib.pyplot as plt
import seaborn as sns


grid_size = 256
file_num = 92

np_filepath = os.path.join(os.path.dirname(__file__), "mean_mat.npy")

# rot_folders = sorted(
#     os.listdir(
#         r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated"
#     ))

img_real_dir = f"/home/proj_depo/docker/data/einzelzahn/images/7a9a625/rgb/{grid_size}x{grid_size}/img"

img_gen_dir = f"/home/proj_depo/docker/data/einzelzahn/gen"

img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:file_num]
real_labels = ["real"] * len(img_real_paths)
img_gen_paths = sorted(glob.glob(os.path.join(img_gen_dir, "*.png")))[:file_num]
gen_labels = ["gen"] * len(img_gen_paths)

img_paths = img_real_paths + img_gen_paths

images = np.asarray(PIL.Image.open(img_paths[0]))[np.newaxis, :]

for img_path in img_real_paths[1:]:
    img = np.asarray(PIL.Image.open(img_path))[np.newaxis, :]
    images = np.concatenate([images, img], axis=0)

for img_path in img_gen_paths:
    img = np.asarray(PIL.Image.open(img_path).convert("RGB"))[np.newaxis, :]
    images = np.concatenate([images, img], axis=0)

img_reals = images[:10]
img_fakes = images[92:][:10]

# print(.shape)
img_reals = img_reals.reshape(-1, 3, 256, 256)
img_fakes = img_fakes.reshape(-1, 3, 256, 256)

tflib.init_tf()
with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    feature_net = pickle.load(f)

num_gpus = 1
max_reals = 92
print(img_reals[0].shape)

feat_real = feature_net.run(img_reals, assume_frozen=True)
feat_fake = feature_net.run(img_fakes, assume_frozen=True)

print(feat_real.shape)
print(feat_fake.shape)
images = np.concatenate([feat_real, feat_fake], axis=0)

# Show correlation structure
print("Starting corr-print")
# Calculate correlation matrix
corr_mat = np.corrcoef(images)

# Plot correlation matrix
plt.figure()
plt.imshow(corr_mat)
plt.colorbar()
# Plot cluster map
sns.clustermap(corr_mat, cmap='viridis', figsize=(8, 8))
plt.show()
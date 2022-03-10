import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from hashlib import sha256
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_paths import import_p_paths
import gan_tools.get_min_metric as gmm
import gan_tools.gan_eval as gev
import plt_tools.plt_creator as pc
# from stylegan2_ada_bra.generate import generate_images
from stylegan2_ada_bra.generate_bra_gpu import init_network, generate_image
from stylegan2_ada_bra.projector_bra import project_nosave

# Set ENVARS for CPU:XLA
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"]="--xla_hlo_profile"

# Specify Seed
seed = 12345
noise_var_seed = 0

save_images = True



grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
network_pkl = "network-snapshot-005456"

# Paths
dirs, paths = import_p_paths(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl, filepath=__file__)



data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(dirs["p_img_gen_script_dir"], exist_ok=True)

# Create unique hash for npz file
network_hash = gev.network_hash(stylegan_folder=stylegan_folder, kimg_str=kimg_str, run_folder=run_folder, network_pkl=network_pkl)

# Generate latent vector from seed
rnd = np.random.RandomState(seed)
z = rnd.randn(1, 512) # [minibatch, component]
# Define latent shape[1] for different grids
shape_dict = {256: 14, 512:16, 1024:18}

noise_vars_values = np.tile(np.random.RandomState(noise_var_seed).randn(1,1,512), (1,shape_dict[grid_size]-1,1))

Gs, noise_vars =  init_network(network_pkl=network_pkl_path)
## Compare
img_dict = {}

img_dict["image_gen_seed"] = generate_image(Gs, noise_vars, seed=seed, img_as_pil=True, noise_var_seed=noise_var_seed)
img_dict["image_gen_z"] = generate_image(Gs, noise_vars, z=z, img_as_pil=True, noise_var_seed=noise_var_seed)

img_dict["image_gen_seed_nonoise"] = generate_image(Gs, seed=seed, img_as_pil=True)
img_dict["image_gen_z_nonoise"] = generate_image(Gs, z=z, img_as_pil=True)

w = np.concatenate([z[np.newaxis, :, :], noise_vars_values], axis=1)
img_dict["image_gen_wz_noise"] = generate_image(Gs, dlatents=w, img_as_pil=True)

w = np.concatenate([z[np.newaxis, :, :], np.zeros(shape=(1,shape_dict[grid_size]-1, 512))], axis=1)
img_dict["image_gen_wz_zeros"] = generate_image(Gs, dlatents=w, img_as_pil=True)

if save_images:
    for key, item in img_dict.items():
        if key.startswith("image"):
            item.save(os.path.join(img_dir, f"seed{seed}-noisevarseed{noise_var_seed}-{key}.png"))
    shutil.copy(__file__, os.path.join(img_dir, os.path.basename(__file__)) )
import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from hashlib import sha256
import shutil
import PIL

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_dir_path import import_dir_path
import gan_tools.get_min_metric as gmm
import gan_tools.gan_eval as gev
import plt_tools.plt_creator as pc
import img_tools.image_processing as ip
# from stylegan2_ada_bra.generate import generate_images
from stylegan2_ada_bra.generate_bra_gpu import init_network, generate_image
from stylegan2_ada_bra.projector_bra import project_nosave

# Set ENVARS for CPU:XLA
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"]="--xla_hlo_profile"

# Specify Seed
seed_init = 12345
save_images = True

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
network_pkl = "network-snapshot-005456"

# Paths
dirs, paths = import_dir_path(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl, grid=grid, filepath=__file__)

# Create unique hash for npz file
network_hash = gev.network_hash(stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl)

# Define latent shape[1] for different grids
shape_dict = {256: 14, 512:16, 1024:18}

Gs =  init_network(network_pkl=paths["network_pkl_path"], nonoise=True)

# images = []
# for ctr  in range(6):
#     images.append(generate_image(Gs, seed= ctr, truncation_psi=None, img_as_pil=False))
# ip.image_grid(images, 2, 3).save("image.png")

## Compare
img_dict = {}
imgs = []
seed_num = 7
trunc_min = -2
trunc_max = 2
trunc_step = 0.2
trunc_range = np.arange(trunc_min,trunc_max+trunc_step,trunc_step)

for seed in range(seed_init,seed_init+seed_num):
  
    # Generate latent vector from seed
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, 512) # [minibatch, component]

    for truncation_psi in trunc_range:
        print(f"Truncation-Psi: {truncation_psi:.1f}")
        img_dict[f"image_gen_z-truncx10_{int(truncation_psi*10)}"] = generate_image(Gs, z=z, truncation_psi=truncation_psi, img_as_pil=True)
        imgs.append(img_dict[f"image_gen_z-truncx10_{int(truncation_psi*10)}"])

      
    if save_images:
        for direc in [dirs["p_img_gen_script_dir"], dirs["p_script_img_gen_dir"]]:
            os.makedirs(direc, exist_ok=True)
            for key, item in img_dict.items():
                if key.startswith("image"):
                    item.save(os.path.join(direc, f"seed{seed}-{key}.png"))
            shutil.copy(__file__, os.path.join(direc, os.path.basename(__file__)) )

grid = ip.image_grid(imgs, rows=seed_num, cols=trunc_range.shape[0]) 
if save_images:
    for direc in [dirs["p_img_gen_script_dir"], dirs["p_script_img_gen_dir"]]:
        os.makedirs(direc, exist_ok=True)
        grid.save(os.path.join(direc, f"grid_seed{seed_init}-{seed}-truncsweep_LR_{int(trunc_min)}{int(trunc_max)}.png"))

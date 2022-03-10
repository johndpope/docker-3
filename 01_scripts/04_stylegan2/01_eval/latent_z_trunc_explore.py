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

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

## Compare
img_dict = {}
imgs = []
seed_num = 7
for seed in range(seed_init,seed_init+seed_num):
  
    # Generate latent vector from seed
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, 512) # [minibatch, component]

    for ctr in range(11):
        truncation_psi = float(ctr)/10
        print(f"Truncation-Psi: {truncation_psi:.1f}")
        img_dict[f"image_gen_z-truncx10_{ctr}"] = generate_image(Gs, z=z, truncation_psi=truncation_psi, img_as_pil=True)
        imgs.append(img_dict[f"image_gen_z-truncx10_{ctr}"])

      
    if save_images:
        for direc in [dirs["p_img_gen_script_dir"], dirs["p_script_img_gen_dir"]]:
            os.makedirs(direc, exist_ok=True)
            for key, item in img_dict.items():
                if key.startswith("image"):
                    item.save(os.path.join(direc, f"seed{seed}-{key}.png"))
            shutil.copy(__file__, os.path.join(direc, os.path.basename(__file__)) )

grid = image_grid(imgs, rows=seed_num, cols=ctr+1) 
if save_images:
    for direc in [dirs["p_img_gen_script_dir"], dirs["p_script_img_gen_dir"]]:
        os.makedirs(direc, exist_ok=True)
        grid.save(os.path.join(direc, f"grid_seed{seed_init}-{seed}-truncsweep_LR_01.png"))

# def save_image_grid(images, filename, drange, grid_size):
#     lo, hi = drange
#     gw, gh = grid_size
#     images = np.asarray(images, dtype=np.float32)
#     images = (images - lo) * (255 / (hi - lo))
#     images = np.rint(images).clip(0, 255).astype(np.uint8)
#     _N, C, H, W = images.shape
#     images = images.reshape(gh, gw, C, H, W)
#     images = images.transpose(0, 3, 1, 4, 2)
#     if C == 3:
#         images = images.reshape(gh * H, gw * W, C)
#         PIL.Image.fromarray(images, 'RGB').save(filename)
#     else:
#         images = images.reshape(gh * H, gw * W)
#         PIL.Image.fromarray(images, 'L').save(filename)  
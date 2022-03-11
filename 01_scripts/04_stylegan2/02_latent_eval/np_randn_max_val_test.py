from curses.ascii import GS
import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from hashlib import sha256
import shutil
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_dir_path import import_dir_path
from stylegan2_ada_bra.generate_bra_gpu import init_network, generate_image

# Set ENVARS for CPU:XLAs
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
os.environ["XLA_FLAGS"]="--xla_hlo_profile"


plot_bool = False

# Paths

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
network_pkl = "network-snapshot-005456"

kimg_num = int([param.split("kimg")[-1] for param in run_folder.split("-") if "kimg" in param][0])
kimg_str = f"kimg{kimg_num:04d}"

# Define latent shape[1] for different grids
shape_dict = {256: 14, 512:16, 1024:18}

# ------------------------------------------------------- #

# Paths
dirs, paths = import_dir_path(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl, grid=grid, filepath=__file__)
# os.makedirs(dirs["p_script_img_gen_dir"], exist_ok=True)

max_val = 0

# Generate latent vector from seed
for seed in tqdm(range(int(1e6))):
    rnd = np.random.RandomState(seed)
    dlatents = rnd.randn(1, shape_dict[grid_size], 512)
    max_val = dlatents.max() if max_val < dlatents.max() else max_val
    if not seed % int(1e5):
        print(max_val)
   
print(max_val)


# Gs =  init_network(network_pkl=paths["network_pkl_path"])
# for ctr in range(0,15):
#     dlatents = rnd.randn(1, shape_dict[grid_size], 512)
#     if ctr>0:
#         dlatents = np.concatenate([dlatents[:,:ctr,:], np.zeros_like(dlatents)[:,ctr:,:]], axis=1)
#     if ctr == shape_dict[grid_size]:
#         dlatents = np.zeros_like(dlatents)

#     img_gen = generate_image(Gs, dlatents=dlatents, img_as_pil=True)

#     img_gen.save(os.path.join(dirs["p_script_img_gen_dir"], f"latent-img_seed{seed}_{ctr}-zeros.png"))

# generate_image(Gs, seed=4, truncation_psi=0,  img_as_pil=True).save(os.path.join(dirs["p_script_img_gen_dir"], f"img_seed{seed}_trunc0.png"))

# shutil.copy(__file__, os.path.join(dirs["p_script_img_gen_dir"], os.path.basename(__file__)))


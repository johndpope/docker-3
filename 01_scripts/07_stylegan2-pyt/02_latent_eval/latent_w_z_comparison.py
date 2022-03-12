import pickle
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
import img_tools.image_processing as ip

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

# seed = 0
# rnd = np.random.RandomState(seed)
# dlatents = rnd.randn(1, shape_dict[grid_size], 512)

z_init = np.ones((1,512))
num_min = 0
num_max = 110
it_step = 10
z_it = np.arange(num_min, num_max+1, it_step)

Gs =  init_network(network_pkl=paths["network_pkl_path"])

os.makedirs(dirs["p_script_img_gen_dir"], exist_ok=True)
# Generate images without truncation and noisevars
# Latent vec z is 512 x same number

imgs = []
for ctr in z_it:

    if ctr == 0:
        z = z_init
    else:    
        z = z_init*ctr

    img = generate_image(Gs, z=z, truncation_psi=None, img_as_pil=True)
    img.save(os.path.join(dirs["p_script_img_gen_dir"], f"img-z-only{int(z[0,0])}.png"))
    imgs.append(img)
    
assert not z_it.shape[0]%2

grid_img = ip.image_grid(imgs=imgs, rows=2, cols=int(z_it.shape[0]/2) )
grid_img.save(os.path.join(dirs["p_script_img_gen_dir"], f"img-z-only1-{num_max}.png"))


linspace_min = 0
linspace_max = 1
z_init = np.linspace(linspace_min,linspace_max,512)[np.newaxis, :]
imgs = []
for ctr in z_it:
    if ctr == 0:
        fac = 1
    else:    
        fac = ctr
    
    z = z_init*fac

    img = generate_image(Gs, z=z, truncation_psi=None, img_as_pil=True)
    img.save(os.path.join(dirs["p_script_img_gen_dir"], f"img-z-linspace-{linspace_min}-{linspace_max}_x{fac}.png"))
    imgs.append(img)
    
assert not z_it.shape[0]%2

grid_img = ip.image_grid(imgs=imgs, rows=2, cols=int(z_it.shape[0]/2) )
grid_img.save(os.path.join(dirs["p_script_img_gen_dir"], f"img-z-linspace-{linspace_min}-{linspace_max}-x1-{num_max}.png"))




shutil.copy(__file__, os.path.join(dirs["p_script_img_gen_dir"], os.path.basename(__file__)))


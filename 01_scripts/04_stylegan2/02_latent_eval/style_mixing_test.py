import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import shutil
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
import sklearn.preprocessing as skp
import scipy
from hashlib import sha256

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import pcd_tools.data_processing as dp
from os_tools.import_paths import import_p_paths
import dnnlib
import dnnlib.tflib as tflib
import gan_tools.gan_eval as gev
import gan_tools.get_min_metric as gmm
import img_tools.image_processing as ip
from stylegan2_ada_bra.style_mixing import style_mixing_example

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})

# Paths
p_style_dir_base, p_img_dir_base, p_latent_dir_base, p_cfg_dir = import_p_paths()

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
kimg = 10000
kimg_str = f"kimg{kimg:04d}"
snapshot_name = "network-snapshot-005456"

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
p_img_dir = os.path.join(p_img_dir_base, "images-style-mixing", grid, stylegan_folder, kimg_str, run_folder, snapshot_name, "img")
os.makedirs(p_img_dir, exist_ok=True)

p_run_dir = [x[0] for x in os.walk(os.path.join(p_style_dir_base, stylegan_folder)) if os.path.basename(x[0]) == run_folder][0]

network_pkl = os.path.join(p_run_dir, f"{snapshot_name}.pkl")

truncation_psi = 0.5
col_styles = list(range(5))
col_seeds = list(range(10))
row_seeds = list(range(1000, 1010))
outdir = p_img_dir

style_mixing_example(network_pkl=network_pkl, row_seeds=row_seeds, col_seeds=col_seeds, truncation_psi=truncation_psi, col_styles=col_styles, outdir=outdir, minibatch_size=4)



# parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
# parser.add_argument('--rows', dest='row_seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
# parser.add_argument('--cols', dest='col_seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
# parser.add_argument('--styles', dest='col_styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
# parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
# parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')










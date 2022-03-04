import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules", "stylegan2_ada_bra"))
from stylegan2_ada_bra.calc_metrics import calc_metrics
import dnnlib.util as util

"""
available metrics:

  ADA paper:
    fid50k_full  Frechet inception distance against the full dataset.
    kid50k_full  Kernel inception distance against the full dataset.
    pr50k3_full  Precision and recall againt the full dataset.
    is50k        Inception score for CIFAR-10.

  Legacy: StyleGAN2
    fid50k       Frechet inception distance against 50k real images.
    kid50k       Kernel inception distance against 50k real images.
    pr50k3       Precision and recall against 50k real images.
    ppl2_wend    Perceptual path length in W at path endpoints against full image.

  Legacy: StyleGAN
    ppl_zfull    Perceptual path length in Z for full paths against cropped image.
    ppl_wfull    Perceptual path length in W for full paths against cropped image.
    ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
    ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    ls           Linear separability with respect to CelebA attributes.
"""
dry_run = False
gpus = 2
metrics = ["fid50k_full", "kid50k_full", "pr50k3_full", "is50k", "ppl2_wend", "ppl_zfull", "ppl_wfull"]


p_base_path = "/home/proj_depo/docker/models/stylegan2/"

default_folder = None #"220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])
kimg = 10000
kimg = f"kimg{kimg:04d}"

if not dry_run:
    # Set ENVARS for CPU:XLA
    os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
    os.environ["XLA_FLAGS"]="--xla_hlo_profile"

if util.ask_yes_no(f"Use last-folder: {last_folder} "):
    folder = last_folder
elif default_folder is not None: 
    if util.ask_yes_no(f"Use default-folder: {default_folder} "):
        folder = default_folder
else:
    folder = str(
        input(
            "Input folder-name to use: \n"
        ))
if not folder:
    raise ValueError("foldername is empty")

p_results_abspath = os.path.join(p_base_path, folder, "results", kimg)

run_dirs = glob.glob(os.path.join(p_results_abspath, "*"))

network_pkls = []
for run_dir in run_dirs:
    network_pkls.extend(glob.glob(os.path.join(run_dir, "*.pkl")))

if not dry_run:
    for network_pkl in network_pkls:
        calc_metrics(network_pkl=network_pkl, metric_names=metrics, metricdata=None, mirror=None, gpus=gpus)

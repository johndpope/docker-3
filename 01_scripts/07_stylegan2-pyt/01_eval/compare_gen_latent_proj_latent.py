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
seed = 12345
plot_bool = False

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
network_pkl = "network-snapshot-005456"
kimg_num = int([param.split("kimg")[-1] for param in run_folder.split("-") if "kimg" in param][0])
kimg_str = f"kimg{kimg_num:04d}"

# Paths
dirs, paths = import_dir_path(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl, grid=grid, filepath=__file__)

# Create unique hash for npz file
network_hash = gev.network_hash(stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=network_pkl)

# Generate latent vector from seed
rnd = np.random.RandomState(seed)
z = rnd.randn(1, 512) # [minibatch, component]

# Save Paths
npz_filepath_data = os.path.join(dirs["p_script_data_dir"], f"seed{seed}_{network_hash}.npz")

if not os.path.exists(npz_filepath_data):
    # Gs, Gs_kwargs, label =  init_network(network_pkl=network_pkl_path, seed_gen=True)
    Gs, _ =  init_network(network_pkl=paths["network_pkl_path"])

    img_gen = generate_image(Gs, seed=seed, img_as_pil=True)

    img_dict = project_nosave(network_pkl=paths["network_pkl_path"], target_pil_image=img_gen)

    img_dict["image_proj_gen_full"] = generate_image(Gs, dlatents=img_dict["dlatents"][-1], img_as_pil=True)

    img_dict["dlatents_zerofill"] = np.concatenate([img_dict["dlatents"][-1][:,:1,:], np.zeros_like(img_dict["dlatents"][-1])[:,1:,:]], axis=1)
    img_dict["image_proj_gen_zerofill"] = generate_image(Gs, dlatents=img_dict["dlatents_zerofill"], img_as_pil=True)

    # Save images
    for direc in [dirs["p_img_gen_script_dir"], dirs["p_script_img_gen_dir"]]:
        os.makedirs(direc, exist_ok=True)
        for key, item in img_dict.items():
            if key.startswith("image"):
                item.save(os.path.join(direc, f"seed{seed}-{key}.png"))
        shutil.copy(__file__, os.path.join(direc, os.path.basename(__file__)) )
    os.makedirs(os.path.dirname(npz_filepath_data), exist_ok=True)
    np.savez(npz_filepath_data, **img_dict)

else:
    # Load the npz file
    print(f"Loading from:\n{npz_filepath_data}")
    img_dict = np.load(npz_filepath_data)

#Plot the optimization process
if plot_bool:
    pc.plot_metrics(   x=np.arange(len(img_dict["dist"])), 
                    y=np.array(img_dict["dist"]), 
                    legend_name=None, 
                    xlabel="Steps",
                    ylabel="Projector Distance", 
                    fig_folder=f"projector_dist_gen_proj_seed{seed}" , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg_str, 
                    run_folder=run_folder, 
                    network_pkl=network_pkl,
                    grid_size=grid_size,
                    master_filepath=__file__,
                    normalize_y=False
                    )

    pc.plot_metrics(   x=np.arange(len(img_dict["loss"])), 
                y=np.array(img_dict["loss"]), 
                legend_name=None, 
                xlabel="Steps",
                ylabel="Projector Loss", 
                fig_folder=f"projector_loss_gen_proj_seed{seed}" , 
                fig_base_dir = dirs["p_script_figure_dir"],
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg_str, 
                run_folder=run_folder, 
                network_pkl=network_pkl,
                grid_size=grid_size,
                master_filepath=__file__,
                normalize_y=False
                )


dlatents_end_full = img_dict["dlatents"][-1].reshape(512,-1)
latent_inner_rank = np.linalg.matrix_rank(dlatents_end_full)

print(f"Projector dlatents Rank: {latent_inner_rank}")


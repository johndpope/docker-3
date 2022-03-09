import pickle
from unicodedata import normalize
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
from hashlib import sha256

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_paths import import_p_paths
import gan_tools.get_min_metric as gmm
import plt_tools.plt_creator as pc

exclude_snap0 = True
# Paths
p_style_dir_base, p_img_dir_base, p_latent_dir_base, p_cfg_dir = import_p_paths()

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"
run_folder = "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgc-bcr-resumecustom-freezed0"
kimg = [param for param in run_folder.split("-") if "kimg" in param][0]

# Create directory for created files
data_dir = os.path.join(os.path.dirname(__file__), "data")
figure_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

p_run_dir = [x[0] for x in os.walk(os.path.join(p_style_dir_base, stylegan_folder)) if os.path.basename(x[0]) == run_folder][0]

metric_types = [metric_file.split("metric-")[-1].split(".")[0] for metric_file in os.listdir(p_run_dir) if "metric" in metric_file]
metrics_dict = {}

for metric_type in metric_types:
    _, _, _, metrics = gmm.get_min_metric(p_run_dir=p_run_dir, metric_type=metric_type)
    metrics_dict[metric_type] = np.array(metrics[1:] if exclude_snap0 else metrics)

snapshot_dir = os.path.join(p_latent_dir_base, image_folder, grid, stylegan_folder, run_folder)

snapshots = sorted(os.listdir(snapshot_dir))
if exclude_snap0:
    snapshots = snapshots[1:] # Exclude snapshot 0

img_names_residual = [name for name in sorted(os.listdir(os.path.join(snapshot_dir, snapshots[0], "latent"))) if "residual" in name]
img_names_train= [name for name in sorted(os.listdir(os.path.join(snapshot_dir, snapshots[0], "latent"))) if not "residual" in name]
img_names_all = sorted(os.listdir(os.path.join(snapshot_dir, snapshots[0], "latent")))
# img_names = [img_names[-2]]

snapshot_kimg = np.array([int(snapshot.split("-")[-1]) for snapshot in snapshots])[:, np.newaxis]

dlatents_arr = np.empty(shape=(0, 512))
img_proj_paths = []
img_target_paths = []
dist_loss_paths = []
dist_losses = []

latent_data = {}

for img_name in img_names_residual:
    latent_data[img_name] = {}
    latent_data[img_name]["dist"] = np.empty(shape=(0,))
    latent_data[img_name]["loss"] = np.empty(shape=(0,))
    latent_data[img_name]["snapshots"] = snapshot_kimg

    for snapshot in snapshots:
        latent_data[img_name][snapshot] = {}
        img_proj_path=os.path.join(snapshot_dir, snapshot, "latent", img_name, "proj.png")
        img_target_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "target.png")
        dlatents_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dlatents.npz")
        dist_loss_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dist_loss.npz")
        latent_data[img_name]["dist"] = np.append(latent_data[img_name]["dist"], np.load(dist_loss_path)["dist"][0])
        latent_data[img_name]["loss"] = np.append(latent_data[img_name]["loss"], np.load(dist_loss_path)["loss"])
        dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)
        latent_data[img_name][snapshot] = {"img_proj_path": img_proj_path, "img_target_path": img_target_path, "dlatents_path": dlatents_path, "dist_loss_path": dist_loss_path}


# Plot kid and fid with single images
dist_list = []
loss_list = []
# for name, item in latent_data.items():
for img_name in img_names_residual:
    fig_obj = plt.figure()
    dist_list.append(latent_data[img_name]["dist"])
    loss_list.append(latent_data[img_name]["loss"])
    img_number = int(img_name.split("_")[1])

    pc.plot_metrics(   x=snapshot_kimg, 
                    y=[np.array(latent_data[img_name]["dist"]), metrics_dict["kid50k_full"], metrics_dict["fid50k_full"]], 
                    legend_name=[f"Projector Distance: Image {img_number}", "Kernel Inception Distance", "Frechet Inception Distance"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metrics", 
                    fig_folder="-".join([f"projector_dist_{img_name}", "kid50k_full", "fid50k_full"])  , 
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__
                )
# Create distance array and calculate the mean across all images
loss_arr = np.array(loss_list)
loss_mean = np.mean(loss_arr, axis=0)

dist_arr = np.array(dist_list)
dist_mean = np.mean(dist_arr, axis=0)

## Plot kid and fid
pc.plot_metrics(   x=snapshot_kimg, 
                y=[dist_mean, metrics_dict["kid50k_full"], metrics_dict["fid50k_full"]], 
                legend_name=["Projector Mean-Distance", "Kernel Inception Distance", "Frechet Inception Distance"], 
                xlabel="Number of k-images",
                ylabel="Normalized metrics", 
                fig_folder="-".join(["projector_dist_mean", "kid50k_full", "fid50k_full"]) , 
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg, 
                run_folder=run_folder, 
                grid_size=grid_size,
                master_filepath=__file__
                )
kid_fid_loss_mean_norm = np.mean(np.concatenate([pc.normalize_01(metrics_dict["fid50k_full"])[:, np.newaxis], pc.normalize_01(metrics_dict["kid50k_full"])[:, np.newaxis], pc.normalize_01(loss_mean)[:, np.newaxis]], axis=1), axis=1)
kid_fid_dist_mean_norm = np.mean(np.concatenate([pc.normalize_01(metrics_dict["fid50k_full"])[:, np.newaxis], pc.normalize_01(metrics_dict["kid50k_full"])[:, np.newaxis], pc.normalize_01(dist_mean)[:, np.newaxis]], axis=1), axis=1)
## Plot kid and fid and dist_mean norm mean
pc.plot_metrics(   x=snapshot_kimg, 
                y=kid_fid_dist_mean_norm, 
                legend_name=["mean(norm(Projector Mean-Dist, KID, FID))"], 
                xlabel="Number of k-images",
                ylabel="Normalized metric", 
                fig_folder="kid_fid_dist_mean_norm" , 
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg, 
                run_folder=run_folder, 
                grid_size=grid_size,
                master_filepath=__file__
                )

## Plot ppls 
pc.plot_metrics(   x=snapshot_kimg, 
                y=[dist_mean, metrics_dict["ppl_wfull"], metrics_dict["ppl_zfull"]], 
                legend_name=["Projector Mean-Distance", "Perceptual Path Length (w)", "Perceptual Path Length (z)"], 
                xlabel="Number of k-images",
                ylabel="Normalized metrics", 
                fig_folder="-".join(["projector_dist_mean", "ppl_wfull", "ppl_zfull"]) , 
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg, 
                run_folder=run_folder, 
                grid_size=grid_size,
                master_filepath=__file__
                )

## Plot ppl2_wend
pc.plot_metrics(   x=snapshot_kimg, 
                y=[dist_mean, metrics_dict["ppl2_wend"]], 
                legend_name=["Projector Mean-Distance", "Perceptual Path Length Endpoints(w)"], 
                xlabel="Number of k-images",
                ylabel="Normalized metrics", 
                fig_folder="-".join(["projector_dist_mean", "ppl2_wend"]) , 
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg, 
                run_folder=run_folder, 
                grid_size=grid_size,
                master_filepath=__file__
                )

txt_name = f"metric_snapshots_min_max-dist-{stylegan_folder.split('_')[0]}_im-{image_folder.split('-')[1]}_{kimg}_{run_folder.split('-')[0]}.txt"
with open(os.path.join(figure_dir, txt_name), "w") as f:
    f.write(f"grid_size: {grid_size}\n")
    f.write(f"image_folder: {image_folder}\n")
    f.write(f"stylegan_folder: {stylegan_folder}\n")
    f.write(f"run_folder: {run_folder}\n\n")

    f.write(f"Snapshot with min(kid_fid_dist_mean_norm) = {kid_fid_dist_mean_norm.min():.4f}: {snapshots[np.argmin(kid_fid_dist_mean_norm)]} \n")
    f.write(f"Snapshot with max(kid_fid_dist_mean_norm) = {kid_fid_dist_mean_norm.max():.4f}: {snapshots[np.argmax(kid_fid_dist_mean_norm)]} \n")

    f.write(f"Snapshot with min(dist_mean) = {dist_mean.min():.4f}: {snapshots[np.argmin(dist_mean)]} \n")
    f.write(f"Snapshot with max(dist_mean) = {dist_mean.max():.4f}: {snapshots[np.argmax(dist_mean)]} \n")
    for metric_type in metric_types:
        f.write(f"Snapshot with min({metric_type}) = {metrics_dict[metric_type].min():.4f}: {snapshots[np.argmin(metrics_dict[metric_type])]}\n")
        f.write(f"Snapshot with max({metric_type}) = {metrics_dict[metric_type].max():.4f}: {snapshots[np.argmax(metrics_dict[metric_type])]}\n")

print(f"Snapshot with min(kid_fid_dist_mean_norm) = {kid_fid_dist_mean_norm.min():.4f}: {snapshots[np.argmin(kid_fid_dist_mean_norm)]}")
print(f"Snapshot with max(kid_fid_dist_mean_norm) = {kid_fid_dist_mean_norm.max():.4f}: {snapshots[np.argmax(kid_fid_dist_mean_norm)]}")

print(f"Snapshot with min(dist_mean) = {dist_mean.min():.4f}: {snapshots[np.argmin(dist_mean)]}")
print(f"Snapshot with max(dist_mean) = {dist_mean.max():.4f}: {snapshots[np.argmax(dist_mean)]}")
for metric_type in metric_types:
    print(f"Snapshot with min({metric_type}) = {metrics_dict[metric_type].min():.4f}: {snapshots[np.argmin(metrics_dict[metric_type])]}")
    print(f"Snapshot with max({metric_type}) = {metrics_dict[metric_type].max():.4f}: {snapshots[np.argmax(metrics_dict[metric_type])]}")


snapshot_opt = snapshots[np.argmin(kid_fid_dist_mean_norm)]
snapshot_opt_kimg = int(snapshot.split("-")[-1])
snapshot = snapshot_opt
for img_name in img_names_train:
    latent_data[img_name] = {}
    latent_data[img_name]["dist"] = np.empty(shape=(0,))
    latent_data[img_name]["loss"] = np.empty(shape=(0,))

    latent_data[img_name]["snapshots"] = snapshot_opt_kimg
    
    latent_data[img_name][snapshot] = {}
    img_proj_path=os.path.join(snapshot_dir, snapshot, "latent", img_name, "proj.png")
    img_target_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "target.png")
    dlatents_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dlatents.npz")
    dist_loss_path = os.path.join(snapshot_dir, snapshot, "latent", img_name, "dist_loss.npz")
    latent_data[img_name]["dist"] = np.append(latent_data[img_name]["dist"], np.load(dist_loss_path)["dist"][0])
    latent_data[img_name]["loss"] = np.append(latent_data[img_name]["loss"], np.load(dist_loss_path)["loss"])
    dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)
    latent_data[img_name][snapshot] = {"img_proj_path": img_proj_path, "img_target_path": img_target_path, "dlatents_path": dlatents_path, "dist_loss_path": dist_loss_path}


# Create distance array and calculate the mean across all images
loss_arr_train = np.array([latent_data[img_name]["loss"] for img_name in img_names_train])
loss_arr_plt = np.concatenate([loss_arr_train, loss_arr[:,snapshots.index(snapshot_opt)][:,np.newaxis]], axis=0)

dist_arr_train = np.array([latent_data[img_name]["dist"] for img_name in img_names_train])
dist_arr_plt = np.concatenate([dist_arr_train, dist_arr[:,snapshots.index(snapshot_opt)][:,np.newaxis]], axis=0)
# dist_mean train vs dist_mean_residual
pc.plot_metrics(   x=np.arange(len(img_names_all)), 
                y=dist_arr_plt, 
                legend_name=None, 
                xlabel="Image number",
                ylabel="Projector Distance", 
                fig_folder="all_distances_over_image_number" , 
                stylegan_folder=stylegan_folder, 
                image_folder=image_folder, 
                kimg=kimg, 
                run_folder=run_folder, 
                grid_size=grid_size,
                master_filepath=__file__,
                normalize_y=False,
                vline_value=len(dist_arr_train)
                )
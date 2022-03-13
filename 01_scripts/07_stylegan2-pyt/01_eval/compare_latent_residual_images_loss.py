import numpy as np
import os
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_dir_path import import_dir_path
import gan_tools.get_min_metric as gmm
import plt_tools.plt_creator as pc

exclude_snap0 = True

grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-4e742fa-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced89"
stylegan_folder = "220312_celebahq-res256-mirror-paper256-kimg100000-ada-target0_5_pyt"
run_folder = "00000-img_prep-mirror-paper256-kimg10000-ada-target0.5-bg-resumecustom-freezed0"
kimg = [param for param in run_folder.split("-") if "kimg" in param][0]

dirs, paths = import_dir_path(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=None, grid=grid, filepath=__file__)


# Create directory for created files
os.makedirs(dirs["p_script_figure_dir"], exist_ok=True)

metric_types = [metric_file.split("metric-")[-1].split(".")[0] for metric_file in os.listdir(dirs["p_run_dir"]) if "metric" in metric_file]
metrics_dict = {}

for metric_type in metric_types:
    _, _, _, metrics = gmm.get_min_metric(p_run_dir=dirs["p_run_dir"], metric_type=metric_type, pyt=True)
    metrics_dict[metric_type] = np.array(metrics[1:] if exclude_snap0 else metrics)

snapshots = sorted(os.listdir(dirs["p_latent_snapshot_dir_base"]))
if exclude_snap0:
    snapshots = snapshots[1:] # Exclude snapshot 0

img_names_residual = [name for name in sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent"))) if "residual" in name]
img_names_train= [name for name in sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent"))) if not "residual" in name]
img_names_all = sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent")))
# img_names = [img_names[-2]]

snapshot_kimg = np.array([int(snapshot.split("-")[-1]) for snapshot in snapshots])[:, np.newaxis]

dlatents_arr = np.empty(shape=(0, 512))
img_proj_paths = []
img_target_paths = []
loss_paths = []


latent_data = {}

for img_name in img_names_residual:
    latent_data[img_name] = {}
    latent_data[img_name]["loss"] = np.empty(shape=(0,))
    latent_data[img_name]["snapshots"] = snapshot_kimg

    for snapshot in snapshots:
        latent_data[img_name][snapshot] = {}
        img_proj_path=os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "proj.png")
        img_target_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "target.png")
        dlatents_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "dlatents.npz")
        loss_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "loss.npz")
        latent_data[img_name]["loss"] = np.append(latent_data[img_name]["loss"], np.load(loss_path)["loss"])
        dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)
        latent_data[img_name][snapshot] = {"img_proj_path": img_proj_path, "img_target_path": img_target_path, "dlatents_path": dlatents_path, "loss_path": loss_path}



# Plot kid and fid with single images
loss_list = []
# for name, item in latent_data.items():
for img_name in img_names_residual:
    fig_obj = plt.figure()
    loss_list.append(latent_data[img_name]["loss"])
    img_number = int(img_name.split("_")[1])
    if "kid50k_full" in metric_types and "fid50k_full" in metric_types:
        pc.plot_metrics(   x=snapshot_kimg, 
                        y=[np.array(latent_data[img_name]["loss"]), metrics_dict["kid50k_full"], metrics_dict["fid50k_full"]], 
                        legend_name=[f"Projector Loss: Image {img_number}", "Kernel Inception Distance", "Frechet Inception Distance"], 
                        xlabel="Number of k-images",
                        ylabel="Normalized metrics", 
                        fig_folder="-".join([f"projector_loss_{img_name}", "kid50k_full", "fid50k_full"])  , 
                        fig_base_dir = dirs["p_script_figure_dir"],
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

if "kid50k_full" in metric_types and "fid50k_full" in metric_types:
    ## Plot kid and fid
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=[loss_mean, metrics_dict["kid50k_full"], metrics_dict["fid50k_full"]], 
                    legend_name=["Projector Mean-Loss", "Kernel Inception Distance", "Frechet Inception Distance"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metrics", 
                    fig_folder="-".join(["projector_loss_mean", "kid50k_full", "fid50k_full"]) , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__
                    )
    kid_fid_loss_mean_norm = np.mean(np.concatenate([pc.normalize_01(metrics_dict["fid50k_full"])[:, np.newaxis], pc.normalize_01(metrics_dict["kid50k_full"])[:, np.newaxis], pc.normalize_01(loss_mean)[:, np.newaxis]], axis=1), axis=1)
    ## Plot kid and fid and loss_mean norm mean
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=kid_fid_loss_mean_norm, 
                    legend_name=["mean(norm(Projector Mean-Loss, KID, FID))"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metric", 
                    fig_folder="kid_fid_loss_mean_norm" , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__
                    )

if "ppl_wfull" in metric_types and "ppl_zfull" in metric_types:
    ## Plot ppls 
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=[loss_mean, metrics_dict["ppl_wfull"], metrics_dict["ppl_zfull"]], 
                    legend_name=["Projector Mean-Loss", "Perceptual Path Length (w)", "Perceptual Path Length (z)"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metrics", 
                    fig_folder="-".join(["projector_loss_mean", "ppl_wfull", "ppl_zfull"]) , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__
                    )
if "ppl2_wend" in metric_types:
    ## Plot ppl2_wend
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=[loss_mean, metrics_dict["ppl2_wend"]], 
                    legend_name=["Projector Mean-Loss", "Perceptual Path Length Endpoints(w)"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metrics", 
                    fig_folder="-".join(["projector_loss_mean", "ppl2_wend"]) , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__
                    )
   
txt_name = f"metric_snapshots_min_max-loss-{stylegan_folder.split('_')[0]}_im-{image_folder.split('-')[1]}_{kimg}_{run_folder.split('-')[0]}.txt"
with open(os.path.join(dirs["p_script_figure_dir"], txt_name), "w") as f:
    f.write(f"grid_size: {grid_size}\n")
    f.write(f"image_folder: {image_folder}\n")
    f.write(f"stylegan_folder: {stylegan_folder}\n")
    f.write(f"run_folder: {run_folder}\n\n")
    if "kid50k_full" in metric_types and "fid50k_full" in metric_types:
        f.write(f"Snapshot with min(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.min():.4f}: {snapshots[np.argmin(kid_fid_loss_mean_norm)]} \n")
        f.write(f"Snapshot with max(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.max():.4f}: {snapshots[np.argmax(kid_fid_loss_mean_norm)]} \n")

    f.write(f"Snapshot with min(loss_mean) = {loss_mean.min():.4f}: {snapshots[np.argmin(loss_mean)]} \n")
    f.write(f"Snapshot with max(loss_mean) = {loss_mean.max():.4f}: {snapshots[np.argmax(loss_mean)]} \n")
    for metric_type in metric_types:
        f.write(f"Snapshot with min({metric_type}) = {metrics_dict[metric_type].min():.4f}: {snapshots[np.argmin(metrics_dict[metric_type])]}\n")
        f.write(f"Snapshot with max({metric_type}) = {metrics_dict[metric_type].max():.4f}: {snapshots[np.argmax(metrics_dict[metric_type])]}\n")
if "kid50k_full" in metric_types and "fid50k_full" in metric_types:
    print(f"Snapshot with min(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.min():.4f}: {snapshots[np.argmin(kid_fid_loss_mean_norm)]}")
    print(f"Snapshot with max(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.max():.4f}: {snapshots[np.argmax(kid_fid_loss_mean_norm)]}")

print(f"Snapshot with min(loss_mean) = {loss_mean.min():.4f}: {snapshots[np.argmin(loss_mean)]}")
print(f"Snapshot with max(loss_mean) = {loss_mean.max():.4f}: {snapshots[np.argmax(loss_mean)]}")
for metric_type in metric_types:
    print(f"Snapshot with min({metric_type}) = {metrics_dict[metric_type].min():.4f}: {snapshots[np.argmin(metrics_dict[metric_type])]}")
    print(f"Snapshot with max({metric_type}) = {metrics_dict[metric_type].max():.4f}: {snapshots[np.argmax(metrics_dict[metric_type])]}")


# snapshot_opt = snapshots[np.argmin(kid_fid_loss_mean_norm)]
# snapshot_opt_kimg = int(snapshot.split("-")[-1])
# snapshot = snapshot_opt
# for img_name in img_names_train:
#     latent_data[img_name] = {}
#     latent_data[img_name]["loss"] = np.empty(shape=(0,))

#     latent_data[img_name]["snapshots"] = snapshot_opt_kimg
    
#     latent_data[img_name][snapshot] = {}
#     img_proj_path=os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "proj.png")
#     img_target_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "target.png")
#     dlatents_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "dlatents.npz")
#     loss_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "loss.npz")
#     latent_data[img_name]["loss"] = np.append(latent_data[img_name]["loss"], np.load(loss_path)["loss"])
#     dlatents_arr = np.concatenate([dlatents_arr, np.load(dlatents_path)["dlatents"][0,0,:][np.newaxis, :]], axis=0)
#     latent_data[img_name][snapshot] = {"img_proj_path": img_proj_path, "img_target_path": img_target_path, "dlatents_path": dlatents_path, "loss_path": loss_path}


# # Create distance array and calculate the mean across all images
# loss_arr_train = np.array([latent_data[img_name]["loss"] for img_name in img_names_train])
# loss_arr_plt = np.concatenate([loss_arr_train, loss_arr[:,snapshots.index(snapshot_opt)][:,np.newaxis]], axis=0)


# # dist_mean train vs dist_mean_residual
# pc.plot_metrics(   x=np.arange(len(img_names_all)), 
#                 y=loss_arr_plt, 
#                 legend_name=None, 
#                 xlabel="Image number",
#                 ylabel="Projector Loss", 
#                 fig_folder="all_losses_over_image_number" , 
#                 fig_base_dir = dirs["p_script_figure_dir"],
#                 stylegan_folder=stylegan_folder, 
#                 image_folder=image_folder, 
#                 kimg=kimg, 
#                 run_folder=run_folder, 
#                 grid_size=grid_size,
#                 master_filepath=__file__,
#                 normalize_y=False,
#                 vline_value=len(loss_arr_train)
#                 )
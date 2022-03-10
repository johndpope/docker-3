import numpy as np
import os
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
from os_tools.import_dir_path import import_dir_path
import gan_tools.get_min_metric as gmm
import plt_tools.plt_creator as pc

exclude_snap0 = True

kimg = 500
kimg_str = f"kimg{kimg:04d}"
grid_size = 256
grid = f"{grid_size}x{grid_size}"
image_folder = "images-dc79a37-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced87"
stylegan_folder = "220303_ffhq-res256-mirror-paper256-noaug"

dirs, paths = import_dir_path()
run_folders = sorted(os.listdir(os.path.join(dirs["p_style_dir_base"], stylegan_folder, "results", kimg_str))) # change for all folders

kid_fid_loss_mean_norm_mins = []
kid_loss_mean_norm_mins = []
kid_loss_mean_mins = []
snapshots_opt = []

for run_folder in run_folders:
    kimg = [param for param in run_folder.split("-") if "kimg" in param][0]

    dirs, paths = import_dir_path(image_folder=image_folder, stylegan_folder=stylegan_folder, run_folder=run_folder, network_pkl=None, grid=grid, filepath=__file__)


    # Create directory for created files
    os.makedirs(dirs["p_script_figure_dir"], exist_ok=True)

    metric_types = [metric_file.split("metric-")[-1].split(".")[0] for metric_file in os.listdir(dirs["p_run_dir"]) if "metric" in metric_file ] # change for kid and fid
    metrics_dict = {}

    for metric_type in metric_types:
        _, _, _, metrics = gmm.get_min_metric(p_run_dir=dirs["p_run_dir"], metric_type=metric_type)
        metrics_dict[metric_type] = np.array(metrics[1:] if exclude_snap0 else metrics)


    snapshots = sorted(os.listdir(dirs["p_latent_snapshot_dir_base"]))
    if exclude_snap0:
        snapshots = snapshots[1:] # Exclude snapshot 0

    img_names_residual = [name for name in sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent"))) if "residual" in name]
    img_names_train= [name for name in sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent"))) if not "residual" in name]
    img_names_all = sorted(os.listdir(os.path.join(dirs["p_latent_snapshot_dir_base"], snapshots[0], "latent")))

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
            img_proj_path=os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "proj.png")
            img_target_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "target.png")
            dlatents_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "dlatents.npz")
            dist_loss_path = os.path.join(dirs["p_latent_snapshot_dir_base"], snapshot, "latent", img_name, "dist_loss.npz")
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

    # Create distance array and calculate the mean across all images
    loss_arr = np.array(loss_list)
    loss_mean = np.mean(loss_arr, axis=0)

    dist_arr = np.array(dist_list)
    dist_mean = np.mean(dist_arr, axis=0)

    # ## Plot kid and fid
    # pc.plot_metrics(   x=snapshot_kimg, 
    #                 y=[loss_mean, metrics_dict["kid50k_full"], metrics_dict["fid50k_full"]], 
    #                 legend_name=["Projector Mean-Loss", "Kernel Inception Distance", "Frechet Inception Distance"], 
    #                 xlabel="Number of k-images",
    #                 ylabel="Normalized metrics", 
    #                 fig_folder="-".join(["projector_loss_mean", "kid50k_full", "fid50k_full"]) , 
    #                 fig_base_dir = dirs["p_script_figure_dir"],
    #                 stylegan_folder=stylegan_folder, 
    #                 image_folder=image_folder, 
    #                 kimg=kimg, 
    #                 run_folder=run_folder, 
    #                 grid_size=grid_size,
    #                 master_filepath=__file__
    #                 )
    # kid_fid_loss_mean_norm = np.mean(np.concatenate([pc.normalize_01(metrics_dict["fid50k_full"])[:, np.newaxis], pc.normalize_01(metrics_dict["kid50k_full"])[:, np.newaxis], pc.normalize_01(loss_mean)[:, np.newaxis]], axis=1), axis=1)
    
    # ## Plot kid and fid and loss_mean norm mean
    # pc.plot_metrics(   x=snapshot_kimg, 
    #                 y=kid_fid_loss_mean_norm, 
    #                 legend_name=["mean(norm(Projector Mean-Loss, KID, FID))"], 
    #                 xlabel="Number of k-images",
    #                 ylabel="Normalized metric", 
    #                 fig_folder="kid_fid_loss_mean_norm" , 
    #                 fig_base_dir = dirs["p_script_figure_dir"],
    #                 stylegan_folder=stylegan_folder, 
    #                 image_folder=image_folder, 
    #                 kimg=kimg, 
    #                 run_folder=run_folder, 
    #                 grid_size=grid_size,
    #                 master_filepath=__file__
    #                 )

    kid_loss_mean_norm = np.mean(np.concatenate([pc.normalize_01(metrics_dict["kid50k_full"])[:, np.newaxis], pc.normalize_01(loss_mean)[:, np.newaxis]], axis=1), axis=1)
    ## Plot fid and loss_mean norm mean
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=kid_loss_mean_norm, 
                    legend_name=["mean(norm(Projector Mean-Loss, KID))"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metric", 
                    fig_folder="kid_loss_mean_norm" , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__,
                    show_fig=False
                    )
    kid_loss_mean = np.mean(np.concatenate([metrics_dict["kid50k_full"][:, np.newaxis], loss_mean[:, np.newaxis]], axis=1), axis=1)
    ## Plot fid and loss_mean norm mean
    pc.plot_metrics(   x=snapshot_kimg, 
                    y=kid_loss_mean, 
                    legend_name=["mean(Projector Mean-Loss, KID)"], 
                    xlabel="Number of k-images",
                    ylabel="Normalized metric", 
                    fig_folder="kid_loss_mean" , 
                    fig_base_dir = dirs["p_script_figure_dir"],
                    stylegan_folder=stylegan_folder, 
                    image_folder=image_folder, 
                    kimg=kimg, 
                    run_folder=run_folder, 
                    grid_size=grid_size,
                    master_filepath=__file__,
                    show_fig=False
                    )    

    txt_name = f"metric_snapshots_min_max-loss-{stylegan_folder.split('_')[0]}_im-{image_folder.split('-')[1]}_{kimg}_{run_folder.split('-')[0]}.txt"
    with open(os.path.join(dirs["p_script_figure_dir"], txt_name), "w") as f:
        f.write(f"grid_size: {grid_size}\n")
        f.write(f"image_folder: {image_folder}\n")
        f.write(f"stylegan_folder: {stylegan_folder}\n")
        f.write(f"run_folder: {run_folder}\n\n")

        # f.write(f"Snapshot with min(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.min():.4f}: {snapshots[np.argmin(kid_fid_loss_mean_norm)]} \n")
        # f.write(f"Snapshot with max(kid_fid_loss_mean_norm) = {kid_fid_loss_mean_norm.max():.4f}: {snapshots[np.argmax(kid_fid_loss_mean_norm)]} \n")

        f.write(f"Snapshot with min(kid_loss_mean_norm) = {kid_loss_mean_norm.min():.4f}: {snapshots[np.argmin(kid_loss_mean_norm)]} \n")
        f.write(f"Snapshot with max(kid_loss_mean_norm) = {kid_loss_mean_norm.max():.4f}: {snapshots[np.argmax(kid_loss_mean_norm)]} \n")

        f.write(f"Snapshot with min(loss_mean) = {loss_mean.min():.4f}: {snapshots[np.argmin(loss_mean)]} \n")
        f.write(f"Snapshot with max(loss_mean) = {loss_mean.max():.4f}: {snapshots[np.argmax(loss_mean)]} \n")
        for metric_type in metric_types:
            f.write(f"Snapshot with min({metric_type}) = {metrics_dict[metric_type].min():.4f}: {snapshots[np.argmin(metrics_dict[metric_type])]}\n")
            f.write(f"Snapshot with max({metric_type}) = {metrics_dict[metric_type].max():.4f}: {snapshots[np.argmax(metrics_dict[metric_type])]}\n")

    
    snapshot_opt = snapshots[np.argmin(kid_loss_mean_norm)]
    
    snapshot_opt_kimg = int(snapshot.split("-")[-1])

    # kid_fid_loss_mean_norm_mins.append(kid_fid_loss_mean_norm.min())
    snapshots_opt.append(snapshot_opt)
    kid_loss_mean_norm_mins.append(kid_loss_mean_norm.min())
    kid_loss_mean_mins.append(kid_loss_mean.min())

print(snapshots_opt)
print(kid_loss_mean_mins)
print(f"Opt run_folder: {run_folders[np.argmin(np.array(kid_loss_mean_mins))]}")
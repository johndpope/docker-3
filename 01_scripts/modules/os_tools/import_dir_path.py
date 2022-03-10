import os

def import_dir_path(image_folder=None, stylegan_folder=None, run_folder=None, network_pkl=None, grid=None, filepath=None):
    """
    Returns os specific p-directories in a dict
    """
    if os.name == "nt":
        p_dir_base = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker"
    elif os.name == "posix":
        p_dir_base = "/home/proj_depo/docker"

    dirs = {}
    paths = {}
    dirs["p_dir_base"] = p_dir_base
    dirs["p_data_dir_base"] = os.path.join(p_dir_base, "data")
    dirs["p_style_dir_base"] = os.path.join(p_dir_base, "models", "stylegan2")
    dirs["p_img_dir_base"] = os.path.join(dirs["p_data_dir_base"], "einzelzahn", "images")
    dirs["p_latent_dir_base"] = os.path.join(dirs["p_data_dir_base"], "einzelzahn", "latents")
    dirs["p_cfg_dir"] = os.path.join(dirs["p_data_dir_base"], "einzelzahn", "cfg")
    dirs["p_data_script_dir_base"] = os.path.join(p_dir_base, "data", "scripts")

    if filepath is not None:
        dirs["p_data_script_dir"] = os.path.join(p_dir_base, "data", "scripts", os.path.basename(filepath).split(".")[0])
        dirs["p_script_figure_dir"] = os.path.join( dirs["p_data_script_dir"], "figure")
        dirs["p_script_data_dir"] = os.path.join( dirs["p_data_script_dir"], "data")
        dirs["p_script_img_dir_base"] = os.path.join(dirs["p_data_script_dir"], "images")

        

    if not any(x is None for x in [image_folder, stylegan_folder, run_folder, grid]):

        if not isinstance(grid, str):
            grid = f"{grid}x{grid}"

        kimg_num = int([param.split("kimg")[-1] for param in run_folder.split("-") if "kimg" in param][0])
        kimg_str = f"kimg{kimg_num:04d}"
        dirs["p_run_dir"] = [x[0] for x in os.walk(os.path.join(dirs["p_style_dir_base"], stylegan_folder)) if os.path.basename(x[0]) == run_folder][0]
        dirs["p_latent_snapshot_dir_base"] = os.path.join(dirs["p_latent_dir_base"], image_folder, grid, stylegan_folder, run_folder)

        if network_pkl is not None:
            dirs["p_img_gen_script_dir"] = os.path.join(dirs["p_img_dir_base"], "images-generated", grid, stylegan_folder, kimg_str, run_folder, network_pkl, os.path.basename(filepath).split(".")[0], "img")  
            dirs["p_script_img_gen_dir"] = os.path.join(dirs["p_script_img_dir_base"], "images-generated", grid, stylegan_folder, kimg_str, run_folder, network_pkl, "img")

            dirs["p_img_gen_dir"] = os.path.join(dirs["p_img_dir_base"], "images-generated", grid, stylegan_folder, kimg_str, run_folder, network_pkl, "img") 
            dirs["p_latent_snapshot_dir"] = os.path.join(dirs["p_latent_snapshot_dir_base"], network_pkl, "latent")
            paths["network_pkl_path"] = os.path.join(dirs["p_run_dir"], f"{network_pkl}.pkl")
    else:
        print(f"Only creating base dirs. Check arguments if more dirs are requested.")
    return dirs, paths

    

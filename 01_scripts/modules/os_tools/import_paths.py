import os

def import_p_paths(image_folder=None, stylegan_folder=None, run_folder=None, network_pkl=None, grid=None, filepath=None):
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
    dirs["p_img_dir_base"] = os.path.join(dirs["p_data_dir_base"] "einzelzahn", "images")
    dirs["p_latent_dir_base"] = os.path.join(dirs["p_data_dir_base"]"einzelzahn", "latents")
    dirs["p_cfg_dir"] = os.path.join(dirs["p_data_dir_base"] "einzelzahn", "cfg")
    dirs["p_script_data_dir_base"] = os.path.join(p_dir_base, "data", "script_data")
    if filepath is not None:
        dirs["p_script_data_dir"] = os.path.join(p_dir_base, "data", "script_data", os.path.basename(filepath).split(".")[0])

    if not any(x is None for x in [image_folder, stylegan_folder, run_folder, network_pkl, grid, filepath]):
        if not isinstance(grid, str):
            grid = f"{grid}x{grid}"

        kimg_num = int([param.split("kimg")[-1] for param in run_folder.split("-") if "kimg" in param][0])
        kimg_str = f"kimg{kimg_num:04d}"
        dirs["p_run_dir"] = [x[0] for x in os.walk(os.path.join(dirs["p_style_dir_base"], stylegan_folder)) if os.path.basename(x[0]) == run_folder][0]
        paths["network_pkl_path"] = os.path.join(dirs["p_run_dir"], f"{network_pkl}.pkl")

        dirs["p_latent_snapshot_dir"] = os.path.join(dirs["p_latent_dir_base"], image_folder, grid, stylegan_folder, run_folder)
        paths["dlatents_path"] = os.path.join(dirs["p_latent_snapshot_dir"], network_pkl, "latent", "img_0018_dc79a37", "dlatents.npz")

        # Create directory for created files
        dirs["p_img_gen_script_dir"] = os.path.join(dirs["p_img_dir_base"], "images-generated", grid, stylegan_folder, kimg_str, run_folder, network_pkl, os.path.basename(filepath).split(".")[0], "img")  
        dirs["p_img_gen_dir"] = os.path.join(dirs["p_img_dir_base"], "images-generated", grid, stylegan_folder, kimg_str, run_folder, network_pkl, "img")  
    
    return dirs, paths

if __name__ == "__main__":
    print(import_p_paths())
    

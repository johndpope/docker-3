import os

def import_p_paths():
    """
    Returns os specific directories for:
    
    p_style_dir_base, p_img_dir_base, p_latent_dir_base, p_cfg_dir
    """
    if os.name == "nt":
        p_dir_base = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker"
    elif os.name == "posix":
        p_dir_base = "/home/proj_depo/docker"

    p_style_dir_base = "/home/proj_depo/docker/models/stylegan2/"
    p_img_dir_base = os.path.join(p_dir_base, "data", "einzelzahn", "images")
    p_latent_dir_base = os.path.join(p_dir_base, "data", "einzelzahn", "latents")
    p_cfg_dir = os.path.join(p_dir_base, "data", "einzelzahn", "cfg")
    return p_style_dir_base, p_img_dir_base, p_latent_dir_base, p_cfg_dir

if __name__ == "__main__":
    print(import_p_paths())
    

import glob
import os
import sys
import shutil

# Warning: Deletes Files // Only procede if you know what you are doing
# Comment out the loops to procede
if __name__ == "__main__":

    used_hashes = [
        "2ad6e8e",
        "dc79a37",
        "7054d07",
        "347380e",
        "56fa467",
        "7a9a625"]

    param_cfg_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
    pcd_dir = r"W:\ukr_data\Einzelzaehne_sorted\grid"

    param_paths = glob.glob(os.path.join(param_cfg_dir, "*"))
    pcd_dirs = glob.glob(os.path.join(pcd_dir, "pcd-*"))


    # for param_path in param_paths:
    #     myhash = os.path.basename(param_path).split(".")[0].split("_")[-1]
    #     if myhash not in used_hashes:
    #         os.remove(param_path)
            
    # for pcd_dir in pcd_dirs:
    #     myhash = os.path.basename(pcd_dir).split("-")[1]
    #     if myhash not in used_hashes:
    #         shutil.rmtree(pcd_dir)

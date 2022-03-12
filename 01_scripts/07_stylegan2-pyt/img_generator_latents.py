import glob
import os
import numpy as np

stylegan_path = "/home/home_bra/repo/stylegan2-ada-bugfixes"
p_path_base = "/home/proj_depo/docker/models/stylegan2/"
folder = "211208_tfl_celebahq_256"
results_folder = "00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom"

p_path = os.path.join(p_path_base, folder, "results", results_folder)
proj_path = os.path.join(p_path, "projector_out")

with open(glob.glob(os.path.join(proj_path, "snapshot*"))[0], "r") as f:
    network_pkl_name = f.readline()
network_pkl_path = os.path.join(p_path, f"{network_pkl_name}.pkl")

print(network_pkl_path)

gen_dirs = [x[0] for x in os.walk(proj_path)][1:]
gen_dirs = [os.path.join(proj_path, x) for x in os.listdir(proj_path) if os.path.isdir(os.path.join(proj_path, x))]
print(gen_dirs)

for gen_dir in gen_dirs:
    print(gen_dir)
    print(glob.glob(os.path.join(gen_dir, "*npz")))
    dlatents_path = glob.glob(os.path.join(gen_dir, "*npz"))[0]
    p_out_path = os.path.join(gen_dir, "img_gen")

    if not os.path.exists(p_out_path) or not any(os.scandir(p_out_path)):
        os.system(f'python {os.path.join(stylegan_path, "generate.py")} \
            --outdir={p_out_path} \
            --network={network_pkl_path} \
            --dlatents={dlatents_path}')

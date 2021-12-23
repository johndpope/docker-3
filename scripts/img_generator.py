import glob
import os
import numpy as np

stylegan_path = "/home/stylegan2-ada"
p_path_base = "/home/proj_depo/docker/models/stylegan2/211208_tfl_celebahq_256"
rel_path_results = "results/00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom"

p_path = os.path.join(p_path_base, rel_path_results)
proj_path = os.path.join(p_path, "projector_out")

with open(glob.glob(os.path.join(proj_path, "snapshot*"))[0], "r") as f:
    network_pkl_name = f.readline()

network_pkl_path = os.path.join(p_path, f"{network_pkl_name}.pkl")
print(network_pkl_path)
direcs = [x[0] for x in os.walk(proj_path)][1:]
gen_dir = direcs[0]
print(gen_dir)

dlatents = np.load(glob.glob(os.path.join(gen_dir, "*npz"))[0])["dlatents"]
print(dlatents.shape)

p_out_path = os.path.join(gen_dir, "img_gen")
print(p_out_path)

if not os.path.exists(p_out_path) or not any(os.scandir(p_out_path)):
    os.system(
        f'python {os.path.join(stylegan_path, "generate.py")} \
        --outdir={p_out_path} \
        --network={network_pkl_path} \
        --dlatents={glob.glob(os.path.join(gen_dir, "*npz"))[0]}'
    )


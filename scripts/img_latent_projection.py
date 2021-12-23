import glob
import os

stylegan_path = "/home/stylegan2-ada"
p_path_base = "/home/proj_depo/docker/models/stylegan2/211208_tfl_celebahq_256"
rel_path_results = "results/00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom"
p_path = os.path.join(p_path_base, rel_path_results)
network_pkl_name = "network-snapshot-000409"
network_pkl_path = os.path.join(p_path, f"{network_pkl_name}.pkl")

with open(os.path.join(p_path, "snapshot.txt", "w")) as f:
    f.write(network_pkl_name)

if os.path.exists(os.path.join(p_path_base, "img_path.txt")):
    print("Loading img_path from file..")
    with open(os.path.join(p_path_base, "img_path.txt"), "r") as f:
        img_path = f.readline().replace("img_prep", "img")
else:
    grid_size = input("Gridsize? format: 256x256: ")
    color = input("rgb / greyscale ?")
    img_paths = [
        x[0]
        for x in os.walk(f"/home/proj_depo/docker/data/einzelzahn/images/{color}/{grid_size}/")
        if os.path.basename(x[0]) == "img"
    ]
    if len(img_paths) > 1:
        for num, img in enumerate(img_paths):
            print(f"Index {num}: {os.path.dirname(img)}")
        img_path = img[int(input(f"Enter Index for preferred img-Files: "))]
    else:
        img_path = img_paths

print(f"img_path = {img_path}")

images = glob.glob(os.path.join(img_path, "*.png"))

# image = os.path.join(img_path, "img_54_1d8846b.png")

for image in images:

    p_out_path = os.path.join(
        p_path, "projector_out", os.path.basename(image).split(".")[0]
    )

    if os.path.exists(p_out_path):
        continue

    os.system(
        f'python {os.path.join(stylegan_path, "projector.py")} \
        --outdir={p_out_path} \
        --target={image} \
        --network={network_pkl_path}'
    )

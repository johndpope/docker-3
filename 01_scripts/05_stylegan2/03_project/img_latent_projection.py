import glob
import os

stylegan_path = "/home/home_bra/repo/stylegan2-ada-bugfixes"
p_path_base = "/home/proj_depo/docker/models/stylegan2/220106_ffhq-res256-mirror-paper256-noaug"
rel_path_results = "results/00000-img_prep-mirror-auto8-kimg10000-ada-resumecustom"
# p_path = os.path.join(p_path_base, rel_path_results)
p_path = p_path_base
# network_pkl_name = "network-snapshot-000409"
# network_pkl_path = os.path.join(p_path, f"{network_pkl_name}.pkl")
# network_pkl_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]
network_pkl_paths = ["https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl", "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl", "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl"]

img_paths = sorted(glob.glob("/home/home_bra/bilder/*.jpg"))
img_path = "/home/home_bra/bilder/1024.jpg"

for network_pkl_path in network_pkl_paths:

    outdir = os.path.join(img_path.split(".")[0],  network_pkl_path.split(".")[-2].split("/")[-1])
    print(outdir)
    os.system(
        f'python {os.path.join(stylegan_path, "projector_bra.py")} \
        --outdir={outdir} \
        --target={img_path} \
        --network={network_pkl_path}'
    )

# with open(os.path.join(p_path, "snapshot.txt", "w")) as f:
#     f.write(network_pkl_name)

# if os.path.exists(os.path.join(p_path_base, "img_path.txt")):
#     print("Loading img_path from file..")
#     with open(os.path.join(p_path_base, "img_path.txt"), "r") as f:
#         img_path = f.readline().replace("img_prep", "img")
# else:
#     grid_size = input("Gridsize? format: 256x256: ")
#     color = input("rgb / greyscale ?")
#     img_paths = [
#         x[0]
#         for x in os.walk(f"/home/proj_depo/docker/data/einzelzahn/images/{color}/{grid_size}/")
#         if os.path.basename(x[0]) == "img"
#     ]
#     if len(img_paths) > 1:
#         for num, img in enumerate(img_paths):
#             print(f"Index {num}: {os.path.dirname(img)}")
#         img_path = img[int(input(f"Enter Index for preferred img-Files: "))]
#     else:
#         img_path = img_paths

# print(f"img_path = {img_path}")

# images = glob.glob(os.path.join(img_path, "*.jpg"))

# image = os.path.join(img_path, "img_54_1d8846b.png")

# for image in images:

# p_out_path = os.path.join(
#     p_path, "projector_out", os.path.basename(image).split(".")[0]
# )


# if os.path.exists(p_out_path):
#     continue
# print(img_paths)


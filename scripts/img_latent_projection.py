import glob
import os

stylegan_path = "/home/stylegan2-ada"

img_paths = [
    x[0]
    for x in os.walk("/home/proj_depo/docker/data/einzelzahn/images/rgb/512x512/")
    if os.path.basename(x[0]) == "img"
]

if len(img_paths) > 1:
    for num, img in enumerate(img_paths):
        print(f"Index {num}: {os.path.dirname(img)}")
    img_path = img[int(input(f"Enter Index for preferred img-Files: "))]
else:
    img_path = img_paths[0]

# images = glob.glob(os.path.join(img_path, "*.png"))
# image = images[0]

image = os.path.join(img_path, "img_54_1d8846b.png")

p_path = "/home/proj_depo/docker/models/stylegan2/211210_brecahad-mirror-paper512-ada/results/00002-img_prep-mirror-auto8-kimg1000-ada-resumecustom-freezed0"
network_pkl_name = "network-snapshot-000491"
network_pkl_path = os.path.join(p_path, f"{network_pkl_name}.pkl")
p_out_path = os.path.join(
    p_path, "projector_out", os.path.basename(image).split(".")[0]
)


os.system(
    f'python {os.path.join(stylegan_path, "projector.py")} \
    --outdir={p_out_path} \
    --target={image} \
    --network={network_pkl_path}'
)

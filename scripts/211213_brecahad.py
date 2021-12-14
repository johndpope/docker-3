import os
import glob
from datetime import date

today_var = date.today().strftime("%y%m%d")

stylegan_path = "/home/stylegan2-ada"

img_paths= [x[0] for x in os.walk("/home/proj_depo/docker/data/einzelzahn/images/rgb/512x512/") if os.path.basename(x[0]) == "img_prep"]

if len(img_paths) > 1:
    for num, img in enumerate(img_paths):
        print(f"Index {num}: {os.path.dirname(img)}")
    img_path = img[
        int(input(f"Enter Index for preferred img-Files: "))
    ]
else:
    img_path = img_paths[0]


# os.system("wget -P /home https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/paper-fig11a-small-datasets/brecahad-mirror-paper512-ada.pkl")

# folder = f"{today_var}_brecahad-mirror-paper512-ada"
folder = "211210_brecahad-mirror-paper512-ada"
p_path=os.path.join("/home/proj_depo/docker/models/stylegan2", folder)
p_results = os.path.join(p_path, "results")

# if not os.path.exists(p_results):
#     os.makedirs(p_results)

# os.system(f"mv {glob.glob(os.path.join('/home', '*.pkl'))[0]} {p_path}")

resumefile_path = glob.glob(os.path.join(p_path, "*.pkl"))[0]

with open(os.path.join(p_path, f"cfg_{today_var}.txt"), "w") as f:
    f.write(f"img_path = {img_path} \n")
    f.write(f"file = {os.path.abspath(__file__)} \n")

os.system( 
    f"python {os.path.join(stylegan_path, 'train.py ')} \
    --gpus=8 \
    --resume={resumefile_path} \
    --freezed=0 \
    --snap=10  \
    --data={img_path} \
    --mirror=1 \
    --kimg=1000 \
    --outdir={p_results} \
    --aug=ada"
)


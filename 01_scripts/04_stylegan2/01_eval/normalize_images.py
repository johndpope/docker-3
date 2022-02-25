import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip

dry_run = False

img_base_dir =  r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated"
img_base_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated\256x256\220224_ffhq-res256-mirror-paper256-noaug"

if dry_run:
    print("\n*-----------*")
    print("DRY RUN")
    print("*-----------*\n")

t0 = time.time()

for img_dir in [x[0] for x in os.walk(img_base_dir)]:
    if os.path.basename(img_dir) == "img":
            img_new_dir = os.path.join(os.path.dirname(img_dir), "img_post")
            print(f"\nCreating post-processed images in {img_new_dir} .. \n")
            if not dry_run:
                ip.ImagePostProcessing(img_dir=img_dir, img_new_dir = img_new_dir)

print(f"Elapsed time in seconds: {(time.time()-t0):.3f}s")


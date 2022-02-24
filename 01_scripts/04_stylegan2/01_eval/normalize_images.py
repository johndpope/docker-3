import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip

img_dir =  r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\images-generated\256x256\220222_ffhq-res256-mirror-paper256-noaug\kimg0750\00004-img_prep-mirror-paper256-kimg750-ada-target0.5-bgcfnc-bcr-resumecustom-freezed0\network-snapshot-000418\img"
# img_dir = r"C:\Users\bra45451\Downloads\Neuer Ordner"
# img_new_dir = os.path.join(os.path.dirname(img_dir), "img_post")

# ip.ImagePostProcessing(img_dir=img_dir, img_new_dir = img_new_dir)
t0 = time.time()
ip.ImagePostProcessing(img_dir=img_dir)
print(f"Elapsed time in seconds: {(time.time()-t0):.3f}s")
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))
import pcd_tools.data_processing as dp

img_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\models\stylegan2\211201\img_gen"

# Create RGB folder
# rgb_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\stylegan2\data\211201\img_rgb"
rgb_dir = os.path.join(img_dir, "rgb")
dp.image_conversion_L_RGB(img_dir=img_dir, rgb_dir=rgb_dir)


import os
import glob
import numpy as np

sha = "8dda7490ddf0b04d0f5750cb20d1e53dd9e7e6bad128f2ccc77061aed3b9b3d6"
p_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\stylegan2\data"

img_filepath = None
for file in glob.glob(os.path.join(p_dir, "**/*.png"), recursive=True):
    if sha in file:
        img_filepath = os.path.dirname(file)
        break

if not img_filepath:
    print("true")

print(
    any(
        [
            True
            for file in glob.glob(os.path.join(p_dir, "**/*.png"), recursive=True)
            if sha in file
        ]
    )
)

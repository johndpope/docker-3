import glob
import os
import sys
import shutil

# Warning: Deletes Files // Only procede if you know what you are doing
# Comment out the loops to procede
if __name__ == "__main__":
    img_dirs = [
            x[0] for x in os.walk(
                "/home/proj_depo/docker/data/einzelzahn/images"
            ) if os.path.basename(x[0]) == "img_prep_stg3"
        ]

    # for img_dir in img_dirs:
    #     shutil.rmtree(img_dir)

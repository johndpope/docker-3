import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from gan_tools.get_min_metric import *
import dnnlib.util as util

p_base_path = "/home/proj_depo/docker/models/stylegan2/"

default_folder = None #"220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])
kimg = 750
kimg = f"kimg{kimg:04d}"


if util.ask_yes_no(f"Use last-folder: {last_folder} "):
    folder = last_folder
elif default_folder is not None: 
    if util.ask_yes_no(f"Use default-folder: {default_folder} "):
        folder = default_folder
else:
    folder = str(
        input(
            "Input folder-name to use: \n"
        ))
if not folder:
    raise ValueError("foldername is empty")

p_results_abspath = os.path.join(p_base_path, folder, "results", kimg)

metric_statistics_path = os.path.join(p_base_path, folder, "metric_statistics", f"statistics_{kimg}.csv")
os.makedirs(os.path.dirname(metric_statistics_path), exist_ok=True)

# print(glob.glob(os.path.join(os.path.dirname(p_results_abspath), "*")))

metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_abspath, as_dataframe=True)

idx_list = get_min_metric_idx_from_dir(p_results_dir=p_results_abspath, metric_threshold=0.02)

metric_list.to_csv(metric_statistics_path, sep=";", decimal=',')

print(metric_list)
# print(metric_list[metric_list["metric-min"]<0.02])

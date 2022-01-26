from get_min_metric import *
import dnnlib.util as util

p_base_path = "/home/proj_depo/docker/models/stylegan2/"
default_folder = "220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])

if util.ask_yes_no(f"Use default-folder: {default_folder} "):
    folder = default_folder
elif util.ask_yes_no(f"Use last-folder: {last_folder} "):
    folder = last_folder
else:
    folder = str(
        input(
            "Input folder-name to use (Folder will be created if it doesnt exist!): \n"
        ))
if not folder:
    raise ValueError("foldername is empty")

folder = default_folder

p_results_abspath = os.path.join(p_base_path, folder, "results", "kimg3000")

# print(glob.glob(os.path.join(os.path.dirname(p_results_abspath), "*")))

metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_abspath, as_dataframe=True)

idx_list = get_min_metric_idx_from_dir(p_results_dir=p_results_abspath, metric_threshold=0.02)

print(idx_list)
# print(metric_list)
print(metric_list[metric_list["metric-min"]<0.02])

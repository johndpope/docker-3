from get_min_metric import *

p_path_base = "/home/proj_depo/docker/models/stylegan2/"
folder = "220106_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"

p_results_abspath = os.path.join(p_path_base, folder, "results")

indx_list, folder_list, metric_list = get_min_metric_idx_from_dir(p_results_dir=p_results_abspath, metric_threshold=30)

for indx in indx_list:
    print(folder_list[indx])
    print(metric_list[indx])
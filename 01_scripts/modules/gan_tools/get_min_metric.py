import numpy as np
import glob
import os


def get_min_metric(p_run_dir):
    metric_file = glob.glob(os.path.join(p_run_dir, "metric*.txt"))[0]
    
    # Open file with metrics and save as var
    with open(metric_file, "r") as f:
        textfile = f.readlines()

    # Get all metrics
    metrics = []
    for line in range(len(textfile)):
        metrics.append(
            float(textfile[line].split("_full ")[-1].replace("\n", "")))

    metrics = np.array(metrics)

    # Error if num_snapshots < 2
    if len(metrics) < 2:
        raise ValueError(
            f"Check the content of {metric_file}. Metric files must have at least 2 elements."
        )

    # Calculate the minimal metric in the converging list of metrics
    metric_min = np.min(metrics)
    metric_end = metrics[-1]
    # Get the index for the metric
    snapshot_num = np.where(metrics == metric_min)[0][0]

    # Select the matching snapshot
    snapshot_name = textfile[snapshot_num].split("time")[0].replace(" ", "")

    return snapshot_name, metric_min, metric_end, metrics


def get_min_metric_list_from_dir(p_results_dir: str, sorted_bool=True, as_dataframe=False):
    folder_list = sorted(os.listdir(p_results_dir))
    metric_list = []
    for idx, p_run_folder in enumerate(folder_list):
        snapshot_name, metric_min, metric_end, _ = get_min_metric(
            p_run_dir=os.path.join(p_results_dir, p_run_folder))
        metric_list.append([metric_min, metric_end/metric_min, snapshot_name, p_run_folder, idx])

    metric_list = sorted(metric_list, key=lambda x: x[0]) if sorted_bool else metric_list
    
    if as_dataframe:
        import pandas as pd
        return pd.DataFrame(metric_list, columns = ["metric-min", "metric_end/metric_min", "Snapshot-Min", "Folder", "Folder-Index"]).set_index(keys="Folder-Index", drop=True, inplace=False)

    else:
        return metric_list


def get_min_metric_idx_from_dir(p_results_dir: str, metric_threshold=None, ratio_metric_min_metric_end=1e10):

    metric_list = get_min_metric_list_from_dir(p_results_dir=p_results_dir, as_dataframe=False, sorted_bool=True)
    indx_list = [
        elems[-1] for elems in metric_list
        if elems[0] < metric_threshold and elems[1]<ratio_metric_min_metric_end
    ]
    return indx_list



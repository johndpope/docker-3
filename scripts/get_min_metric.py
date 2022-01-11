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
        metrics.append(float(textfile[line].split("_full ")[-1].replace("\n", "")))

    metrics = np.array(metrics)

    # Calculate the (rolling) difference for the metric
    diff_metrics = np.diff(metrics)

    # Neglects snapshots after certain metric if it diverges (diff > threshold diff)
    threshold_diff = 2
    for ctr, diff_metric in enumerate(diff_metrics):
        diff_num = ctr
        if diff_metric > threshold_diff:
            break

    metrics = metrics[:diff_num + 2]

    # Calculate the minimal metric in the converging list of metrics
    metric_min = np.min(metrics)

    # Get the index for the metric
    snapshot_num = np.where(metrics == metric_min)[0][0]

    # Select the matching snapshot
    snapshot_name = textfile[snapshot_num].split("time")[0].replace(" ", "")

    return snapshot_name, metric_min

def get_min_metric_idx_from_dir(p_results_dir: str, metric_threshold: float):
    results = sorted(os.listdir(p_results_dir))
    metric_list = []
    for p_run_dir in results:
        _, metric_min = get_min_metric(os.path.join(p_results_dir, p_run_dir))
        metric_list.append(metric_min)

    indx_list = [idx for idx, metric_min in enumerate(metric_list) if metric_min < metric_threshold]    
    return indx_list

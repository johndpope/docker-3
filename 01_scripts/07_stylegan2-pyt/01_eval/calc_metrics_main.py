import os
import sys
import glob
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules", "stylegan2_ada_bra"))
# from stylegan2_ada_pytorch_bra.calc_metrics import calc_metrics
import dnnlib.util as util
import load_tools.file_loader as fl

"""
available metrics:

  ADA paper:
    fid50k_full  Frechet inception distance against the full dataset.
    kid50k_full  Kernel inception distance against the full dataset.
    pr50k3_full  Precision and recall againt the full dataset.
    is50k        Inception score for CIFAR-10.

  Legacy: StyleGAN2
    fid50k       Frechet inception distance against 50k real images.
    kid50k       Kernel inception distance against 50k real images.
    pr50k3       Precision and recall against 50k real images.
    ppl2_wend    Perceptual path length in W at path endpoints against full image.

  Legacy: StyleGAN
    ppl_zfull    Perceptual path length in Z for full paths against cropped image.
    ppl_wfull    Perceptual path length in W for full paths against cropped image.
    ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
    ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    ls           Linear separability with respect to CelebA attributes.
"""
stylegan_version = 1
stylegan_versions = ["stylegan2_ada", "stylegan2_ada_pytorch", "stylegan3",]


dry_run = False
gpus = 1
infinity_run = True
sort_metric_files = False
remove_dublicates = False
metrics = ["kid50k_full", "fid50k_full", "ppl2_wend", "ppl_zfull", "ppl_wfull"]

## Parameters for fast processing
reverse_metrics = False
reverse_snapshots = False
reverse_run_dirs = False

p_base_path = "/home/proj_depo/docker/models/stylegan2/"
## Paths
stylegan_path = f"/home/home_bra/01_scripts/modules/{stylegan_versions[stylegan_version]}_bra"

default_folder = None #"220118_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
run_folder =  "00001-img_prep-mirror-paper256-kimg10000-ada-target0.5-bgcfnc-resumecustom-freezed0"         # Set to None if all


last_folder = os.path.basename(sorted(os.listdir(p_base_path))[-1])
kimg = 10000
kimg = f"kimg{kimg:04d}"

if not dry_run:
    # Set ENVARS for CPU:XLA
    os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
    os.environ["XLA_FLAGS"]="--xla_hlo_profile"

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

if run_folder is None:
    run_dirs = glob.glob(os.path.join(p_results_abspath, "*"))
else:
    run_dirs = [os.path.join(p_results_abspath, run_folder)]

if reverse_run_dirs:
    run_dirs = run_dirs[::-1]






while 1:
    network_pkls = []
    for run_dir in run_dirs:
        network_pkls_single = glob.glob(os.path.join(run_dir, "*.pkl"))
        if reverse_snapshots:
            network_pkls_single=network_pkls_single[::-1]
        network_pkls.extend(network_pkls_single)

    if reverse_metrics:
        metrics=metrics[::-1]

    if not dry_run:
        for metric in metrics:
            if network_pkls:
                for network_pkl in network_pkls:
                    print("\n --------------- \n")
                    print(f"Metric: {metric}")
                    print(f"Snapshot: {os.path.basename(network_pkl).split('.')[0]}")
                    print(f"Directory: \n{os.path.dirname(network_pkl)}")
                    print("\n")

                    metric_file = os.path.join(os.path.dirname(network_pkl), f'metric-{metric}.jsonl')
                    textfile=None
                    if os.path.exists(metric_file):
                        textfile = fl.load_jsonl(metric_file)

                    if textfile is None or not any( [os.path.basename(network_pkl) == dict(textline)["snapshot_pkl"] for textline in textfile]):
                        for ctr in range(2):
                            try:
                                os.system(f'python {os.path.join(stylegan_path, "calc_metrics.py")} \
                                    --metrics={metric} \
                                    --network={network_pkl} \
                                    --gpus={gpus}')
                                # calc_metrics(network_pkl=network_pkl, metric_names=[metric], mdata=None, mirror=None, gpus=gpus)
                                break
                            except Exception as e:
                                print(e)
                                time.sleep(60)

    
    if not infinity_run:
        break

#     # Decorate with metadata.
#     return dnnlib.EasyDict(
#         results         = dnnlib.EasyDict(results),
#         metric          = metric,
#         total_time      = total_time,
#         total_time_str  = dnnlib.util.format_time(total_time),
#         num_gpus        = opts.num_gpus,
#     )

# #----------------------------------------------------------------------------

# def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
#     metric = result_dict['metric']
#     assert is_valid_metric(metric)
#     if run_dir is not None and snapshot_pkl is not None:
#         snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

#     jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
#     print(jsonl_line)
#     if run_dir is not None and os.path.isdir(run_dir):
#         with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
#             f.write(jsonl_line + '\n')

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import PIL

import data_processing as dp
from generate_bra import init_network, generate_image


# @st.experimental_singleton(suppress_st_warning=True)
@st.cache(suppress_st_warning=True)
def init_func():
    ## Initial
    t0 = time.time()
    current_path = os.path.dirname(__file__)
    img_dir = os.path.join(current_path, "images")
    os.makedirs(img_dir, exist_ok=True)

    cfg_search_dir = "/home/home_bra/ukr_data/Einzelzaehne_sorted/grid"

    p_path_base = "/home/proj_depo/docker/models/stylegan2/"
    folder = "211231_brecahad-mirror-paper512-ada"
    results_folder = "00000-img_prep-stylegan2-kimg3000-ada-resumecustom-freezed0"

    p_path = os.path.join(p_path_base, folder, "results", results_folder)
    metric_file = glob.glob(os.path.join(p_path, "metric*.txt"))[0]

    # Open file with metrics and save as var
    with open(metric_file, "r") as f:
        textfile = f.readlines()

    # Get all metrics
    metrics = []
    for line in range(len(textfile)):
        metrics.append(
            float(textfile[line].split("_full ")[-1].replace("\n", "")))

    metrics = np.array(metrics)

    # Calculate the (rolling) difference for the metric
    diff_metrics = np.diff(metrics)

    # Neglect snapshots after certain metric if it diverges (diff > threshold diff)
    threshold_diff = 2
    for ctr, diff_metric in enumerate(diff_metrics):
        diff_num = ctr
        if diff_metric > threshold_diff:
            print(diff_num)
            break

    metrics = metrics[:diff_num + 2]

    # Calculate the minimal metric in the converging list of metrics
    metric_min = np.min(metrics)

    # Get the index for the metric
    snapshot_num = np.where(metrics == metric_min)[0][0]

    # Select the matching snapshot
    snapshot_name = textfile[snapshot_num].split("time")[0].replace(" ", "")
    network_pkl_path = glob.glob(os.path.join(p_path, f"{snapshot_name}*"))[0]

    st.write(f"Metric: {metric_min}")
    st.write(f"Snapshot: {snapshot_name}")

    st.write(f"Elapsed time in seconds (Init): {(time.time()-t0):.3f}s")
    
    return current_path, img_dir, cfg_search_dir, network_pkl_path

# @st.experimental_singleton
def init_network_func(network_pkl_path):
    t1 = time.time()
    Gs = init_network(network_pkl=network_pkl_path)
    # st.write(f"Elapsed time in seconds (Init network): {(time.time()-t1):.3f}s")
    return Gs

current_path, img_dir, cfg_search_dir, network_pkl_path = init_func()


print(network_pkl_path)
# seed = 3
# seed = int(st.number_input("Seed: ", min_value=0, step=1))
seed = int(st.sidebar.number_input("Seed: ", min_value=0, step=1))
img_path = os.path.join(img_dir, f"seed{seed:04d}.png")
print(img_path)

t2 = time.time()

if not os.path.exists(img_path):
    Gs = init_network_func(network_pkl_path=network_pkl_path)
    img_raw = generate_image(Gs=Gs, seed=seed, outdir=img_dir)
    img = PIL.Image.fromarray(img_raw, 'RGB')
else:
    print("Loading from image..")
    img = PIL.Image.open(img_path)


show_image = st.button("Show Image")
if show_image:
    st.image(img)


st.write(f"Elapsed time in seconds (Generate): {(time.time()-t2):.3f}s")

pcd_arr = dp.img_to_pcd_single(img_path,
                               z_crop=0.2,
                               cfg_search_dir=cfg_search_dir)

# Sort the pcd arr along z axis for correct colors in 3d-Plot (z ascending)
pcd_arr = pcd_arr[pcd_arr[:, 2].argsort(), :]

fig2 = go.Figure(
    go.Scatter3d(x=pcd_arr[:, 0],
                 y=pcd_arr[:, 1],
                 z=pcd_arr[:, 2],
                 mode='markers',
                 marker=dict(size=3)))

figs = [fig2]

color_array = (pcd_arr[:, 2] - pcd_arr[:, 2].min()) / (
    pcd_arr[:, 2].max() - pcd_arr[:, 2].min()) * 255

for fig in figs:
    fig.update_layout(width=800, height=800)
    fig.update_scenes(xaxis_visible=False,
                      yaxis_visible=False,
                      zaxis_visible=False)
    # fig.update_traces(surfaceaxis=2, surfacecolor="rgb(200,200,200)", marker=dict(color="rgb(200,200,200"), selector=dict(type='scatter3d'))
    # fig.update_traces(marker=dict(
    #     color=color_array,
    #     cmin=0,
    #     cmax=255,
    # ),
    #                   selector=dict(type='scatter3d'))
    fig.update_traces(marker=dict(color=color_array, cmin=0, cmax=255),
                      selector=dict(type='scatter3d'))

    st.plotly_chart(fig)

# for seed in range(1000):
#     if not os.path.exists(os.path.join(img_dir, f"seed{seed:04d}.png")):
#         generate_image(Gs=Gs, seed=seed, outdir=img_dir)

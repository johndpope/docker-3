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
from generate_bra import *

import tensorflow as tf

st.session_state.page_config = st.set_page_config(
    page_title="Tooth-Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

t00 = time.time()
placeholder = st.empty()

def init():
    ## Initial
    t0 = time.time()
    current_path = os.path.dirname(__file__)
    img_dir = os.path.join(current_path, "images")
    os.makedirs(img_dir, exist_ok=True)
    external_url = "http://172.20.30.156:8501/"
    print(f"External URL: {external_url}")

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

    print(f"Metric: {metric_min}")
    print(f"Snapshot: {snapshot_name}")
    print(f"Elapsed time in seconds (Init): {(time.time()-t0):.3f}s")

    st.session_state.network_pkl_path = network_pkl_path
    st.session_state.img_dir = img_dir
    st.session_state.cfg_search_dir = cfg_search_dir
    st.session_state.snapshot_num = snapshot_num


def show_image():
    st.image(PIL.Image.fromarray(st.session_state["img"], 'RGB'))


# st.checkbox("Show Image", on_change=show_image)


def init_network():
    t1 = time.time()
    Gs, Gs_kwargs, label = init_network2(
        network_pkl=st.session_state.network_pkl_path,
        outdir=st.session_state.img_dir)
    st.session_state.session = tf.get_default_session()
    st.session_state.Gs = Gs
    st.session_state.Gs_kwargs = Gs_kwargs
    st.session_state.label = label
    print(f"Elapsed time in seconds (Init network): {(time.time()-t1):.3f}s")


def generate():
    placeholder.text("Generating..")

    img_path = os.path.join(st.session_state.img_dir,
                            f"seed{int(st.session_state.seed):04d}.png")

    if not os.path.exists(img_path):
        t2 = time.time()

        with st.session_state.session.as_default():
            img = generate_image2(Gs=st.session_state.Gs,
                                  seed=int(st.session_state.seed),
                                  outdir=None,
                                  Gs_kwargs=st.session_state.Gs_kwargs,
                                  label=st.session_state.label)

        # print(f"Elapsed time in seconds (Generate): {(time.time()-t2):.3f}s")
        # img = PIL.Image.fromarray(img_raw, 'RGB')

    else:
        print("Loading from image..")
        img = np.asarray(PIL.Image.open(img_path))

    pcd_arr = dp.img_to_pcd_single(
        img=img, z_crop=0.2, cfg_search_dir=st.session_state.cfg_search_dir)

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
        fig.update_layout(width=1000, height=1000)
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

        # st.plotly_chart(fig)

    # for seed in range(1000):
    #     if not os.path.exists(os.path.join(img_dir, f"seed{seed:04d}.png")):
    #         generate_image(Gs=Gs, seed=seed, outdir=img_dir)

    st.session_state["fig"] = fig
    st.session_state["img"] = img
    placeholder.empty()


def change_snapshot():
    if "snapshot_user" in st.session_state:
        st.session_state.network_pkl_path = os.path.join(
            os.path.dirname(st.session_state.network_pkl_path),
            os.path.basename(st.session_state.snapshot_user))
        init_network()
        generate()


## Init
if "cfg_search_dir" not in st.session_state: 
    init()

if 'Gs' not in st.session_state:
    placeholder.text("Initializing..")
    init_network()
    placeholder.empty()

st.sidebar.text(
    f"Current Snapshot:  \n{os.path.basename(st.session_state.network_pkl_path)}"
)

print(st.session_state.network_pkl_path)
st.sidebar.selectbox("Choose a snapshot", [
    os.path.basename(snapshot) for snapshot in glob.glob(
        os.path.join(os.path.dirname(st.session_state.network_pkl_path),
                     "*.pkl"))
],
                     index=int(st.session_state.snapshot_num),
                     key="snapshot_user",
                     on_change=change_snapshot)
print(st.session_state.snapshot_user)

st.sidebar.number_input("Choose a input-seed: ",
                        min_value=0,
                        step=1,
                        key="seed",
                        on_change=generate)

if "fig" not in st.session_state:
    generate()

st.plotly_chart(st.session_state["fig"], use_container_width=True)

st.sidebar.text(f"Elapsed time in seconds:  \n{(time.time()-t00):.3f}s")

if st.sidebar.button("Delete Cache"):
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]

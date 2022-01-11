from matplotlib.pyplot import autoscale
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
import generate_bra_v2 as gen

import tensorflow as tf

st.session_state.page_config = st.set_page_config(
    page_title="Tooth-Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

title = "Tooth-Generator"
# st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
st.header(title)

## Functions 

def init():
    ## Initial
    delete_cache()
    st.session_state.init = False
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
    snapshot_opt_num = np.where(metrics == metric_min)[0][0]

    # Select the matching snapshot
    snapshot_name = textfile[snapshot_opt_num].split("time")[0].replace(" ", "")
    network_pkl_path = glob.glob(os.path.join(p_path, f"{snapshot_name}*"))[0]

    print(f"Metric: {metric_min}")
    print(f"Snapshot: {snapshot_name}")
    print(f"Elapsed time in seconds (Init): {(time.time()-t0):.3f}s")

    st.session_state.network_pkl_path = network_pkl_path
    st.session_state.img_dir = img_dir
    st.session_state.cfg_search_dir = cfg_search_dir
    st.session_state.snapshot_opt_num = snapshot_opt_num
    st.session_state.init = True


def init_network():
    t1 = time.time()
    Gs, Gs_kwargs, label = gen.init_network(
        network_pkl=st.session_state.network_pkl_path,
        outdir=st.session_state.img_dir)
    st.session_state.session = tf.get_default_session()
    st.session_state.Gs = Gs
    st.session_state.Gs_kwargs = Gs_kwargs
    st.session_state.label = label
    st.session_state.init_network = True
    print(f"Elapsed time in seconds (Init network): {(time.time()-t1):.3f}s")


def generate():
    placeholder.text("Generating..")

    img_path = os.path.join(st.session_state.img_dir,
                            f"seed{int(st.session_state.seed):04d}.png")

    if not os.path.exists(img_path):
        t2 = time.time()

        with st.session_state.session.as_default():
            img = gen.generate_image(Gs=st.session_state.Gs,
                                  seed=int(st.session_state.seed),
                                  outdir=None,
                                  Gs_kwargs=st.session_state.Gs_kwargs,
                                  label=st.session_state.label)

        print(f"Elapsed time in seconds (Generate): {(time.time()-t2):.3f}s")
    else:
        print("Loading from image..")
        img = np.asarray(PIL.Image.open(img_path))

    pcd_arr = dp.img_to_pcd_single(
        img=img, z_crop=0.2, cfg_search_dir=st.session_state.cfg_search_dir)

    # Sort the pcd arr along z axis for correct colors in 3d-Plot (z ascending)
    pcd_arr = pcd_arr[pcd_arr[:, 2].argsort(), :]

    fig = go.Figure(
        go.Scatter3d(x=pcd_arr[:, 0],
                     y=pcd_arr[:, 1],
                     z=pcd_arr[:, 2],
                     mode='markers',
                     marker=dict(size=3)))

    color_array = (pcd_arr[:, 2] - pcd_arr[:, 2].min()) / (
        pcd_arr[:, 2].max() - pcd_arr[:, 2].min()) * 255

    fig.update_layout(height=1000,scene_camera=dict(eye=dict(x=0., y=0., z=2.))) # width=1000, height=1000, 
    fig.update_scenes(xaxis_visible=False,
                        yaxis_visible=False,
                        zaxis_visible=False)
                        
    fig.update_traces(marker=dict(color=color_array, cmin=0, cmax=255),
                        selector=dict(type='scatter3d'))

    st.session_state.fig = fig
    st.session_state.img = img
    st.session_state.channels = st.session_state.img.shape[2]
    placeholder.empty()


def change_snapshot():
    if "snapshot_user" in st.session_state:
        st.session_state.network_pkl_path = os.path.join(
            os.path.dirname(st.session_state.network_pkl_path),
            os.path.basename(st.session_state.snapshot_user))
        init_network()
        generate()


def delete_cache():      
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]


# ## Init
# col1, col2 = st.columns((2,1))

t00 = time.time()
placeholder = st.empty()

if "init" not in st.session_state: 
    init()
elif not st.session_state.init:
    init()


if 'init_network' not in st.session_state:
    placeholder.text("Initializing..")
    init_network()
    placeholder.empty()

print(st.session_state.network_pkl_path)

with st.sidebar:
    st.title("Settings")
    st.selectbox("Choose a snapshot", [
        os.path.basename(snapshot) for snapshot in glob.glob(
            os.path.join(os.path.dirname(st.session_state.network_pkl_path),
                        "*.pkl"))
    ],
                        index=int(st.session_state.snapshot_opt_num),
                        key="snapshot_user",
                        on_change=change_snapshot)

    st.number_input("Choose a input-seed: ",
                            min_value=0,
                            step=1,
                            key="seed",
                            on_change=generate)

print(st.session_state.snapshot_user)

if "fig" not in st.session_state:
    generate()

# with col1:
#     my_expander_chart = st.expander(label="Show 3D Plot", expanded=True)
#     with my_expander_chart:
#         st.plotly_chart(st.session_state["fig"], use_container_width=True)
# with col2:
#     
#     with my_expander_img:
#         st.image(PIL.Image.fromarray(st.session_state.img, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw model Output")

# with col1:
#     st.plotly_chart(st.session_state["fig"], use_container_width=True)
# with col2:
#     st.image(PIL.Image.fromarray(st.session_state.img, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw model Output")


st.plotly_chart(st.session_state["fig"], use_container_width=True)
with st.sidebar:
    my_expander_img = st.expander(label="Show Image", expanded=False)
    with my_expander_img:
        st.image(PIL.Image.fromarray(st.session_state.img, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw model Output")

with st.sidebar:
    st.text(
        f"Current Snapshot:  \n{os.path.basename(st.session_state.network_pkl_path)}"
    )
    st.text(f"Current Seed:  \n{st.session_state.seed}")

    st.button("Clear Cache and Initialize App", on_click=init)

# st.sidebar.text(f"Elapsed time in seconds:  \n{(time.time()-t00):.3f}s")


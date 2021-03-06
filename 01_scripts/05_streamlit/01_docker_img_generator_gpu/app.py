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
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

import pcd_tools.data_processing as dp
import gan_tools.get_min_metric as gm
import img_tools.image_processing as ip
import stylegan2_ada_bra.generate_bra_gpu as gen

## Functions 

def init():
    ## Initial
    delete_cache()
    st.session_state.init = False
    st.session_state.t0 = time.time()
    current_path = os.path.dirname(__file__)
    img_dir = os.path.join(current_path, "images")
    os.makedirs(img_dir, exist_ok=True)
    external_url = "http://172.20.30.156:8501/"
    print(f"External URL: {external_url}")

    p_path_base = "/home/proj_depo/docker/models/stylegan2/"
    folder = "220303_ffhq-res256-mirror-paper256-noaug" #"220222_ffhq-res256-mirror-paper256-noaug" #"220202_ffhq-res256-mirror-paper256-noaug"#"220118_ffhq-res256-mirror-paper256-noaug" #"220106_ffhq-res256-mirror-paper256-noaug" #"211231_brecahad-mirror-paper512-ada"
    kimg=10000 # as int

    # Get the param hash
    with open(os.path.join(p_path_base, folder, "img_path.txt")) as f:
        img_dir = f.read()

    p_results_dir = os.path.join(p_path_base, folder, "results", f"kimg{kimg:04d}")
    results_folder_list = sorted(os.listdir(p_results_dir))
    # Get sorted metric list as pd dataframe
    metric_list = gm.get_min_metric_list_from_dir(p_results_dir=p_results_dir, sorted_bool=True, as_dataframe=True)

    if "results_folder_user" in st.session_state:
        results_folder = st.session_state.results_folder_user
    else:
        # Get the Folder with the best snapshot metric
        results_folder = metric_list.iloc[0,:]["Folder"]

    st.session_state.results_folder = results_folder

    p_run_dir = os.path.join(p_results_dir, results_folder)

    # Get snapshot name and metric of the best Snapshot
    snapshot_name, metric_min, _, metrics = gm.get_min_metric(p_run_dir=p_run_dir)

    network_pkl_path = glob.glob(os.path.join(p_run_dir, f"{snapshot_name}*"))[0]

    print(f"Metric: {metric_min}")
    print(f"Snapshot: {snapshot_name}")
    print(f"Elapsed time in seconds (Init): {(time.time()-st.session_state.t0):.3f}s")

    st.session_state.network_pkl_path = network_pkl_path
    st.session_state.img_dir = img_dir
    st.session_state.snapshot_name = f"{snapshot_name}-{metric_min}"
    st.session_state.available_snapshots = [
        f"{os.path.basename(snapshot).split('.')[0]}-{metric}" for snapshot, metric in zip(sorted(glob.glob(os.path.join(p_run_dir, "*.pkl"))), metrics)]
    st.session_state.metrics_full = metrics    
    st.session_state.results_folder_list = results_folder_list
    st.session_state.p_results_dir = p_results_dir
    st.session_state.init = True
    st.session_state.init_network = False
    st.session_state.generate_flag =  False


def init_network():
    # Initialize the Conversion Params
    st.session_state.ImageConverterParams = dp.ImageConverterParams(img_dir=st.session_state.img_dir)

    t = time.time()
    Gs = gen.init_network(network_pkl=st.session_state.network_pkl_path, nonoise=True)
    st.session_state.session = tf.get_default_session()
    st.session_state.Gs = Gs
    st.session_state.init_network = True
    st.session_state.generate_flag = False
    print(f"Elapsed time in seconds (Init network): {(time.time()-t):.3f}s")


def generate():
    placeholder.text("Generating..")
    if st.session_state.generate_flag:
        st.session_state.t0 = time.time()
        st.session_state.fig_cache = st.session_state.fig

    img_path = os.path.join(st.session_state.img_dir,
                            f"seed{int(st.session_state.seed):04d}.png")

    if not os.path.exists(img_path):
        t0 = time.time()

        with st.session_state.session.as_default():
            img = gen.generate_image(Gs=st.session_state.Gs,
                                  seed=int(st.session_state.seed))

        print(f"Elapsed time in seconds (Generate): {(time.time()-t0):.3f}s")
    else:
        print("Loading from image..")
        img = np.asarray(PIL.Image.open(img_path))

    t1 = time.time()
    
    ImageConverterSingle = dp.ImageConverterSingle(img=img, rot = False, center = False, crop = True)
    pcd_arr=ImageConverterSingle.img_to_pcd()
    

    print(f"Elapsed time in seconds (img to pcd): {(time.time()-t1):.3f}s")

    # Sort the pcd arr along z axis for correct colors in 3d-Plot (z ascending)
    pcd_arr = pcd_arr[pcd_arr[:, 2].argsort(), :]

    fig = go.Figure(
        go.Scatter3d(x=pcd_arr[:, 0],
                     y=pcd_arr[:, 1],
                     z=pcd_arr[:, 2],
                     mode='markers',
                     marker=dict(size=3)))

    fig.update_layout(height=900)
    # fig.update_layout(scene_camera=dict(eye=dict(x=2., y=2., z=2.)))
    fig.update_scenes(xaxis_visible=False,
                        yaxis_visible=False,
                        zaxis_visible=False)
                        
    fig.update_traces(marker=dict(color=pcd_arr[:, 2], colorscale="Viridis",colorbar=dict(thickness=20, title="height in mm", titleside="bottom")),
                        selector=dict(type='scatter3d'))

    st.session_state.fig = fig
    st.session_state.img = img
    st.session_state.channels = img.shape[2] if img.ndim == 3 else 1
    st.session_state.generate_flag = True
    placeholder.empty()

def change_folder():
    if "results_folder_user" in st.session_state:          
        init()

def change_snapshot():
    if "snapshot_user" in st.session_state:          
        st.session_state.network_pkl_path = os.path.join(
            os.path.dirname(st.session_state.network_pkl_path), f"{'-'.join(st.session_state.snapshot_user.split('-')[:-1])}.pkl" )
        init_network()

def delete_cache():      
    # Delete all the items in Session state
    dont_delete_list = ["page_config", "t0", "results_folder_user", "seed"]
    for key in st.session_state.keys():
        if key not in dont_delete_list:
            del st.session_state[key]
    print("Cache clear")


# ## Init
st.session_state.page_config = st.set_page_config(
    page_title="Tooth-Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

placeholder = st.empty()

if "init" not in st.session_state: 
    init()

if not st.session_state.init_network:
    placeholder.text("Initializing..")
    init_network()
    placeholder.empty()

print(f"Network pkl path: {st.session_state.network_pkl_path}")

with st.sidebar:
    st.title("Tooth-Generator")
    st.header("Settings")

    st.selectbox("Choose a Folder:", st.session_state.results_folder_list,
                        index=st.session_state.results_folder_list.index(st.session_state.results_folder),
                        key="results_folder_user",
                        on_change=change_folder)

    st.selectbox("Choose a Snapshot:", st.session_state.available_snapshots,
                        index=st.session_state.available_snapshots.index(st.session_state.snapshot_name), #int(st.session_state.snapshot_opt_num),
                        key="snapshot_user",
                        on_change=change_snapshot)
    st.text("network-snapshot-<number_of_kimg>-<error_metric>")

    st.number_input("Choose a Input-Seed: ",
                            min_value=0,
                            value = st.session_state.seed if "seed" in st.session_state else 0,
                            step=1,
                            key="seed",
                            on_change=generate)

print(f"Seed: {st.session_state.seed}")

if not st.session_state.generate_flag:
    generate()

st.plotly_chart(st.session_state.fig, use_container_width=True)

if "fig_cache" in st.session_state:
    my_expander_fig = st.expander(label="Show Comparison With Previous Tooth", expanded=False)
    
    with my_expander_fig:
        col1, col2 = st.columns(2)
        with col1:
            st.text("Cached")
            st.text(f"Seed: {st.session_state.seed_cache} ")
            st.text(f"Snapshot: {os.path.basename(st.session_state.network_pkl_path_cache)} ")
            st.plotly_chart(st.session_state.fig_cache, use_container_width=True)
            st.image(PIL.Image.fromarray(st.session_state.img_cache, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw Model Output (Cached)")
        with col2:
            st.text("Current")
            st.text(f"Seed: {st.session_state.seed} ")
            st.text(f"Snapshot: {os.path.basename(st.session_state.network_pkl_path)} ")
            st.plotly_chart(st.session_state.fig, use_container_width=True)
            st.image(PIL.Image.fromarray(st.session_state.img, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw Model Output (Current)")

with st.sidebar:
    my_expander_img = st.expander(label="Show Image", expanded=False)
    with my_expander_img:
        st.image(PIL.Image.fromarray(st.session_state.img, "RGB" if st.session_state.channels == 3 else "L"), use_column_width="always", caption="Raw Model Output")

    st.text(
        f"Current Snapshot:  \n{os.path.basename(st.session_state.network_pkl_path)}"
    )
    st.text(f"Current Seed:  \n{st.session_state.seed}")

    st.button("Clear Cache and Initialize App", on_click=init)

if "seed" in st.session_state:
    st.session_state.seed_cache = st.session_state.seed
if "network_pkl_path" in st.session_state:
    st.session_state.network_pkl_path_cache = st.session_state.network_pkl_path
if "img" in st.session_state:
    st.session_state.img_cache = st.session_state.img

print(f"Elapsed time in seconds (app-run): {(time.time()-st.session_state.t0):.3f}s")
